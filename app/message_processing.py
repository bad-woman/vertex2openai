import base64
import re
import json
import time
import random 
import httpx
import concurrent.futures
from typing import List, Dict, Any, Tuple
import config as app_config

from google.genai import types
from models import OpenAIMessage, ContentPartText, ContentPartImage

import io
try:
    from PIL import Image
except ImportError:
    Image = None

def optimize_image_bytes(image_data: bytes, original_mime: str, max_size_bytes: int = int(1.5 * 1024 * 1024)) -> Tuple[bytes, str]:
    """激进且高画质的输入图片压缩方案：超过1.5MB的图强制限制边长至1536，斩断上下文体积恶性膨胀循环"""
    if Image is None:
        return image_data, original_mime
    
    # 只要在 1.5MB 以内，直接原样放行，确保画质绝对保真
    if len(image_data) <= max_size_bytes:
        return image_data, original_mime
        
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # 抹平透明通道以防 JPEG 转换报错
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA' or img.mode == 'LA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 强势降维：修图参考无需 4K 像素流，1536px 已经超越绝大多数 OCR 和细节识别需求
            max_dim = 1536
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            # 首选 Q=85 极致画质
            img.save(output, format='JPEG', quality=85, optimize=True)
            opt_data = output.getvalue()
            
            # 若由于细节过于繁复依然超重，执行二次压缩降至 Q=70（文件大小将锁定在 500KB 上下）
            if len(opt_data) > max_size_bytes:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=70, optimize=True)
                opt_data = output.getvalue()
                
            return opt_data, "image/jpeg"
    except Exception as e:
        print(f"警告: 输入图片极致压缩失败，已回退为原图传输: {e}")
        return image_data, original_mime

SUPPORTED_ROLES = ["user", "model", "function"] 

def extract_reasoning_by_tags(full_text: str, tag_name: str) -> Tuple[str, str]:
    if not tag_name or not isinstance(full_text, str):
        return "", full_text if isinstance(full_text, str) else ""
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    pattern = re.compile(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", re.DOTALL)
    reasoning_parts = pattern.findall(full_text)
    normal_text = pattern.sub("", full_text)
    reasoning_content = "".join(reasoning_parts)
    return reasoning_content.strip(), normal_text.strip()

def _extract_markdown_images_to_parts(text: str) -> Tuple[List[types.Part], str]:
    parts = []
    remaining_text = text
    pattern = r"!\[[^\]]*\]\(data:(image/[a-zA-Z0-9+.-]+);base64,([a-zA-Z0-9+/=]+)\)"
    matches = list(re.finditer(pattern, text))
    
    if matches:
        for match in reversed(matches):
            mime_type = match.group(1)
            b64_data = match.group(2)
            if not mime_type.startswith("image/"):
                continue
            try:
                raw_bytes = base64.b64decode(b64_data)
                opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                start, end = match.span()
                remaining_text = remaining_text[:start] + remaining_text[end:]
            except Exception as e:
                print(f"Error extracting markdown image: {e}")
        parts.reverse()
    
    remaining_text = re.sub(r"[ \t]+", " ", remaining_text).strip()
    return parts, remaining_text

def create_gemini_prompt(messages: List[OpenAIMessage]) -> List[types.Content]:
    print("Converting OpenAI messages to Gemini format...")
    raw_gemini_messages = []
    for idx, message in enumerate(messages):
        role = message.role
        if role == "system":
            continue

        parts = []
        current_gemini_role = "" 

        if role == "tool":
            tool_call_id_str = message.tool_call_id or ""
            
            if "__thought__" not in tool_call_id_str:
                mock_text = f"[System Observation - Tool '{message.name}' Result]:\n{message.content}"
                parts.append(types.Part.from_text(text=mock_text))
                current_gemini_role = "user"
            else:
                if message.name and message.content is not None:
                    tool_output_data = {}
                    try:
                        if isinstance(message.content, str) and \
                           (message.content.strip().startswith("{") and message.content.strip().endswith("}")) or \
                           (message.content.strip().startswith("[") and message.content.strip().endswith("]")):
                            tool_output_data = json.loads(message.content)
                        else: 
                            tool_output_data = {"result": message.content}
                    except json.JSONDecodeError:
                        tool_output_data = {"result": str(message.content)}

                    parts_id = tool_call_id_str.split("__thought__")
                    real_tool_id = parts_id[0]
                    b64_sig = parts_id[1]
                    thought_sig_bytes = None
                    try: 
                        thought_sig_bytes = base64.b64decode(b64_sig)
                    except: 
                        pass

                    func_resp_kwargs = {"name": message.name, "response": tool_output_data}
                    if real_tool_id:
                        func_resp_kwargs["id"] = real_tool_id
                        
                    try:
                        part_kwargs = {"function_response": types.FunctionResponse(**func_resp_kwargs)}
                        if thought_sig_bytes:
                            part_kwargs["thought_signature"] = thought_sig_bytes
                        resp_part = types.Part(**part_kwargs)
                    except Exception as e:
                        print(f"Warning: Failed to inject FunctionResponse signature: {e}")
                        resp_part = types.Part.from_function_response(name=message.name, response=tool_output_data)

                    parts.append(resp_part)
                    current_gemini_role = "function"
                
        elif role == "assistant" and message.tool_calls:
            current_gemini_role = "model"
            for tool_call in message.tool_calls:
                function_call_data = tool_call.get("function", {})
                function_name = function_call_data.get("name", "unknown")
                arguments_str = function_call_data.get("arguments", "{}")
                tool_call_id_str = tool_call.get("id", "")
                
                if "__thought__" not in tool_call_id_str:
                    mock_text = f"[Model Action Log]: Decided to call function '{function_name}' with arguments: {arguments_str}"
                    parts.append(types.Part.from_text(text=mock_text))
                else:
                    try: 
                        parsed_arguments = json.loads(arguments_str)
                    except json.JSONDecodeError: 
                        parsed_arguments = {} 
                        
                    parts_id = tool_call_id_str.split("__thought__")
                    real_tool_id = parts_id[0]
                    b64_sig = parts_id[1]
                    thought_sig_bytes = None
                    try: 
                        thought_sig_bytes = base64.b64decode(b64_sig)
                    except: 
                        pass

                    fc_kwargs = {"name": function_name, "args": parsed_arguments}
                    if real_tool_id:
                        fc_kwargs["id"] = real_tool_id
                        
                    try:
                        part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                        if thought_sig_bytes:
                            part_kwargs["thought_signature"] = thought_sig_bytes
                        fc_part = types.Part(**part_kwargs)
                    except Exception as e:
                        print(f"Warning: Failed to inject FunctionCall signature: {e}")
                        fc_part = types.Part.from_function_call(name=function_name, args=parsed_arguments)
                        
                    parts.append(fc_part)
                    
            if message.content:
                if isinstance(message.content, str):
                    image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                    if clean_text: parts.append(types.Part.from_text(text=clean_text))
        else: 
            if message.content is None: continue
            
            current_gemini_role = role
            if current_gemini_role == "assistant": current_gemini_role = "model"
            if current_gemini_role not in SUPPORTED_ROLES:
                current_gemini_role = "user"

            if isinstance(message.content, str):
                image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                if clean_text: parts.append(types.Part.from_text(text=clean_text))
                
                if current_gemini_role != "model":
                    parts.extend(image_parts) 
                elif image_parts:
                    parts.append(types.Part.from_text(text="[图片已省略 / Image omitted]"))

            elif isinstance(message.content, list):
                for part_item in message.content:
                    if isinstance(part_item, dict):
                        if part_item.get("type") == "text":
                            text_content = part_item.get("text", "\n")
                            image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
                            if clean_text: parts.append(types.Part.from_text(text=clean_text))
                            
                            if current_gemini_role != "model":
                                parts.extend(image_parts)
                            elif image_parts:
                                parts.append(types.Part.from_text(text="[图片已省略 / Image omitted]"))

                        elif part_item.get("type") == "image_url":
                            if current_gemini_role != "model":
                                image_url = part_item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:"):
                                    mime_match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                                    if mime_match:
                                        mime_type, b64_data = mime_match.groups()
                                        raw_bytes = base64.b64decode(b64_data)
                                        opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                        parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                                elif image_url.startswith("http"):
                                    try:
                                        def fetch_img():
                                            client_args = {"timeout": 10.0, "follow_redirects": True}
                                            if app_config.PROXY_URL:
                                                client_args["proxy"] = app_config.PROXY_URL
                                            if getattr(app_config, "SSL_CERT_FILE", None):
                                                client_args["verify"] = app_config.SSL_CERT_FILE
                                            with httpx.Client(**client_args) as client:
                                                resp = client.get(image_url)
                                                resp.raise_for_status()
                                                return resp.content, resp.headers.get("content-type", "image/jpeg")
                                        with concurrent.futures.ThreadPoolExecutor() as pool:
                                            future = pool.submit(fetch_img)
                                            img_bytes, mime_type = future.result(timeout=12) 
                                            opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                            parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                                    except Exception as e:
                                        print(f"Warning: Failed to fetch remote image {image_url}: {e}")
                            else:
                                parts.append(types.Part.from_text(text="[图片已省略 / Image omitted]"))
                    elif hasattr(part_item, "text"):
                        parts.append(types.Part.from_text(text=part_item.text))
                    
                    elif hasattr(part_item, "type") and getattr(part_item, "type") == "image_url":
                        if current_gemini_role != "model":
                            img_url_data = part_item.image_url
                            url_str = getattr(img_url_data, "url", "") if hasattr(img_url_data, "url") else (img_url_data.get("url", "") if isinstance(img_url_data, dict) else "")
                            
                            if url_str.startswith("data:"):
                                mime_match = re.match(r"data:([^;]+);base64,(.+)", url_str)
                                if mime_match:
                                    mime_type, b64_data = mime_match.groups()
                                    raw_bytes = base64.b64decode(b64_data)
                                    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                    parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                            elif url_str.startswith("http"):
                                try:
                                    def fetch_img():
                                        client_args = {"timeout": 10.0, "follow_redirects": True}
                                        if app_config.PROXY_URL:
                                            client_args["proxy"] = app_config.PROXY_URL
                                        if getattr(app_config, "SSL_CERT_FILE", None):
                                            client_args["verify"] = app_config.SSL_CERT_FILE
                                        with httpx.Client(**client_args) as client:
                                            resp = client.get(url_str)
                                            resp.raise_for_status()
                                            return resp.content, resp.headers.get("content-type", "image/jpeg")
                                    with concurrent.futures.ThreadPoolExecutor() as pool:
                                        future = pool.submit(fetch_img)
                                        img_bytes, mime_type = future.result(timeout=12) 
                                        opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                        parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                                except Exception as e:
                                    print(f"Warning: Failed to fetch remote image {url_str}: {e}")
                        else:
                            parts.append(types.Part.from_text(text="[图片已省略 / Image omitted]"))

        if not parts: continue
        raw_gemini_messages.append(types.Content(role=current_gemini_role, parts=parts))

    merged_messages = []
    for msg in raw_gemini_messages:
        if merged_messages and merged_messages[-1].role == msg.role:
            merged_messages[-1].parts.append(types.Part.from_text(text="\n\n"))
            merged_messages[-1].parts.extend(msg.parts)
        else:
            merged_messages.append(msg)

    if not merged_messages:
        merged_messages.append(types.Content(role="user", parts=[types.Part.from_text(text="继续")]))

    return merged_messages

def _create_safety_ratings_html(safety_ratings: list) -> str:
    if not safety_ratings:
        return ""
    highest_rating = max(safety_ratings, key=lambda r: r.probability_score)
    highest_score = highest_rating.probability_score

    if highest_score <= 0.33: color = "#0f8"  
    elif highest_score <= 0.66: color = "yellow"
    else: color = "#bf555d"

    summary_category = highest_rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
    summary_probability = highest_rating.probability.name
    summary_score_str = f"{highest_rating.probability_score:.7f}" if highest_rating.probability_score is not None else "None"
    summary_severity_str = f"{highest_rating.severity_score:.8f}" if highest_rating.severity_score is not None else "None"
    summary_line = f"{summary_category}: {summary_probability} (Score: {summary_score_str}, Severity: {summary_severity_str})"

    ratings_list = []
    for rating in safety_ratings:
        category = rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
        probability = rating.probability.name
        score_str = f"{rating.probability_score:.7f}" if rating.probability_score is not None else "None"
        severity_str = f"{rating.severity_score:.8f}" if rating.severity_score is not None else "None"
        ratings_list.append(f"{category}: {probability} (Score: {score_str}, Severity: {severity_str})")
    all_ratings_str = "\n".join(ratings_list)

    css_style = "<style>.cb{border:1px solid #444;margin:10px;border-radius:4px;background:#111}.cb summary{padding:8px;cursor:pointer;background:#222}.cb pre{margin:0;padding:10px;border-top:1px solid #444;white-space:pre-wrap}</style>"
    html_output = (
        f"{css_style}"
        f"<details class='cb'>"
        f"<summary style='color:{color}'>{summary_line} ▼</summary>"
        f"<pre>\n--- Safety Ratings ---\n{all_ratings_str}\n</pre>"
        f"</details>"
    )
    return html_output

def _convert_image_to_markdown(image_data: bytes, mime_type: str) -> str:
    try:
        b64_data = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"
        return f"![Image]({data_url})"
    except Exception as e:
        print(f"Error converting image to markdown: {e}")
        return "[Image could not be displayed]"

def parse_gemini_response_for_reasoning_and_content(gemini_response_candidate: Any) -> Tuple[str, str]:
    reasoning_text_parts = []
    normal_text_parts = []
    candidate_part_text = ""
    if hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None:
        candidate_part_text = str(gemini_response_candidate.text)

    gemini_candidate_content = None
    if hasattr(gemini_response_candidate, "content"):
        gemini_candidate_content = gemini_response_candidate.content

    if gemini_candidate_content and hasattr(gemini_candidate_content, "parts") and gemini_candidate_content.parts:
        for part_item in gemini_candidate_content.parts:
            if hasattr(part_item, "function_call") and part_item.function_call is not None: 
                continue
            
            part_text = ""
            if hasattr(part_item, "text") and part_item.text is not None:
                part_text = str(part_item.text)
            elif hasattr(part_item, "inline_data") and part_item.inline_data is not None:
                inline_data = part_item.inline_data
                if hasattr(inline_data, "data") and hasattr(inline_data, "mime_type"):
                    image_bytes = inline_data.data
                    mime_type = inline_data.mime_type
                    part_text = _convert_image_to_markdown(image_bytes, mime_type)
            elif hasattr(part_item, "file_data") and part_item.file_data is not None:
                file_data = part_item.file_data
                if hasattr(file_data, "file_uri"):
                    file_uri = file_data.file_uri
                    part_text = f"![Image]({file_uri})"
            
            part_is_thought = hasattr(part_item, "thought") and part_item.thought is True

            if part_is_thought: reasoning_text_parts.append(part_text)
            elif part_text: normal_text_parts.append(part_text)
            
    elif candidate_part_text: normal_text_parts.append(candidate_part_text)
    elif gemini_candidate_content and hasattr(gemini_candidate_content, "text") and gemini_candidate_content.text is not None:
        normal_text_parts.append(str(gemini_candidate_content.text))
    elif hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None and not gemini_candidate_content: 
        normal_text_parts.append(str(gemini_response_candidate.text))

    return "".join(reasoning_text_parts), "".join(normal_text_parts)

def process_gemini_response_to_openai_dict(gemini_response_obj: Any, request_model_str: str) -> Dict[str, Any]:
    choices = []
    response_timestamp = int(time.time())
    base_id = f"chatcmpl-{response_timestamp}-{random.randint(1000,9999)}"

    if hasattr(gemini_response_obj, "candidates") and gemini_response_obj.candidates:
        for i, candidate in enumerate(gemini_response_obj.candidates):
            message_payload = {"role": "assistant"}
            
            raw_finish_reason = getattr(candidate, "finish_reason", None)
            openai_finish_reason = "stop" 
            if raw_finish_reason:
                if hasattr(raw_finish_reason, "name"): raw_finish_reason_str = raw_finish_reason.name.upper()
                else: raw_finish_reason_str = str(raw_finish_reason).upper()

                if raw_finish_reason_str == "STOP": openai_finish_reason = "stop"
                elif raw_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
                elif raw_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
                elif raw_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"
            
            function_call_detected = False
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call is not None: 
                        fc = part.function_call
                        
                        real_id = getattr(fc, "id", None)
                        if not real_id: real_id = getattr(fc, "thought_signature", None)
                        
                        thought_sig = getattr(part, "thought_signature", None)
                        thought_sig_b64 = ""
                        if thought_sig:
                            if isinstance(thought_sig, bytes):
                                thought_sig_b64 = base64.b64encode(thought_sig).decode("utf-8")
                            elif isinstance(thought_sig, str):
                                thought_sig_b64 = thought_sig

                        safe_name = fc.name.replace(" ", "_")
                        rand_num = int(time.time() * 10000 + random.randint(0, 9999))

                        if real_id:
                            if thought_sig_b64:
                                tool_call_id = f"{real_id}__thought__{thought_sig_b64}"
                            else:
                                tool_call_id = real_id
                        else:
                            if thought_sig_b64:
                                tool_call_id = f"call_{base_id}_{i}_{safe_name}__thought__{thought_sig_b64}"
                            else:
                                tool_call_id = f"call_{base_id}_{i}_{safe_name}_{rand_num}"
                        
                        if "tool_calls" not in message_payload:
                            message_payload["tool_calls"] = []
                        
                        message_payload["tool_calls"].append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args or {})
                            }
                        })
                        message_payload["content"] = None 
                        openai_finish_reason = "tool_calls" 
                        function_call_detected = True
            
            if not function_call_detected:
                reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
                if app_config.SAFETY_SCORE and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    safety_html = _create_safety_ratings_html(candidate.safety_ratings)
                    if reasoning_str: reasoning_str += safety_html
                    else: normal_content_str += safety_html
                
                message_payload["content"] = normal_content_str
                if reasoning_str: message_payload["reasoning_content"] = reasoning_str
            
            choice_item = {"index": i, "message": message_payload, "finish_reason": openai_finish_reason}
            if hasattr(candidate, "logprobs") and candidate.logprobs is not None: choice_item["logprobs"] = candidate.logprobs
            choices.append(choice_item)
            
    elif hasattr(gemini_response_obj, "text") and gemini_response_obj.text is not None:
         content_str = gemini_response_obj.text or ""
         choices.append({"index": 0, "message": {"role": "assistant", "content": content_str}, "finish_reason": "stop"})
    else: 
         choices.append({"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"})

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(gemini_response_obj, "usage_metadata"):
        um = gemini_response_obj.usage_metadata
        if hasattr(um, "prompt_token_count"): usage_data["prompt_tokens"] = um.prompt_token_count
        if hasattr(um, "candidates_token_count"):
            usage_data["completion_tokens"] = um.candidates_token_count
            if hasattr(um, "total_token_count"): usage_data["total_tokens"] = um.total_token_count
            else: usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
        elif hasattr(um, "total_token_count"): 
             usage_data["total_tokens"] = um.total_token_count
             if usage_data["prompt_tokens"] > 0 and usage_data["total_tokens"] > usage_data["prompt_tokens"]:
                 usage_data["completion_tokens"] = usage_data["total_tokens"] - usage_data["prompt_tokens"]
        else: usage_data["total_tokens"] = usage_data["prompt_tokens"] 

    return {
        "id": base_id, "object": "chat.completion", "created": response_timestamp,
        "model": request_model_str, "choices": choices,
        "usage": usage_data
    }

def convert_to_openai_format(gemini_response: Any, model: str) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(gemini_response, model)