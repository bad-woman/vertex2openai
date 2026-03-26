import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types
from google import genai

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import (
    create_gemini_prompt,
    create_encrypted_gemini_prompt,
    create_encrypted_full_gemini_prompt,
    ENCRYPTION_INSTRUCTIONS,
)
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
from project_id_discovery import discover_project_id

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        
        OPENAI_DIRECT_SUFFIX = "-openai"
        OPENAI_SEARCH_SUFFIX = "-openaisearch"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " 
        
        base_model_name = request.model 
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        is_openai_direct_model = False
        is_openai_search_model = False
        
        if base_model_name.endswith(OPENAI_SEARCH_SUFFIX):
            is_openai_search_model = True
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_SEARCH_SUFFIX)]
        elif base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_DIRECT_SUFFIX)]
            
        if EXPERIMENTAL_MARKER in base_model_name:
            is_openai_direct_model = True

        is_auto_model = base_model_name.endswith("-auto")
        if is_auto_model: base_model_name = base_model_name[:-len("-auto")]

        is_grounded_search = base_model_name.endswith("-search")
        if is_grounded_search: base_model_name = base_model_name[:-len("-search")]

        is_encrypted_full_model = base_model_name.endswith("-encrypt-full")
        if is_encrypted_full_model: base_model_name = base_model_name[:-len("-encrypt-full")]

        is_encrypted_model = base_model_name.endswith("-encrypt")
        if is_encrypted_model: base_model_name = base_model_name[:-len("-encrypt")]

        is_nothinking_model = base_model_name.endswith("-nothinking")
        if is_nothinking_model: base_model_name = base_model_name[:-len("-nothinking")]

        is_max_thinking_model = base_model_name.endswith("-max")
        if is_max_thinking_model: base_model_name = base_model_name[:-len("-max")]

        # ==========================================
        # 核心：智能识别 image 并强制拦截
        # ==========================================
        is_image_model = "image" in request.model.lower()
        if is_image_model:
            base_model_name = base_model_name.replace("-image", "").replace("_image", "")
            is_openai_direct_model = False

        gen_config_dict = create_generation_config(request)

        is_thinking_capable = "gemini-2.5" in base_model_name or "gemini-3" in base_model_name
        is_lite_model = "flash-lite" in base_model_name

        if is_thinking_capable:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        if is_lite_model or is_image_model:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = False

        client_to_use = None

        if is_express_model_request:
            if express_key_manager_instance.get_total_keys() == 0:
                error_msg = f"Model '{request.model}' requires an Express API key, but none are configured."
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

            total_keys = express_key_manager_instance.get_total_keys()
            for attempt in range(total_keys):
                key_tuple = express_key_manager_instance.get_express_api_key()
                if key_tuple:
                    original_idx, key_val = key_tuple
                    try:
                        if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                            project_id = await discover_project_id(key_val)
                            base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                            client_to_use = genai.Client(
                                vertexai=True,
                                api_key=key_val,
                                http_options=types.HttpOptions(base_url=base_url)
                            )
                            client_to_use._api_client._http_options.api_version = None
                        else:
                            client_to_use = genai.Client(vertexai=True, api_key=key_val)
                        break 
                    except Exception as e:
                        client_to_use = None 
                else:
                    client_to_use = None

            if client_to_use is None: 
                return JSONResponse(status_code=500, content=create_openai_error_response(500, "All configured Express API keys failed.", "server_error"))
        
        else: 
            rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()
            
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                except Exception as e:
                    return JSONResponse(status_code=500, content=create_openai_error_response(500, str(e), "server_error"))
            else: 
                return JSONResponse(status_code=401, content=create_openai_error_response(401, "No SA credentials available.", "authentication_error"))

        if not is_openai_direct_model and client_to_use is None:
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        if is_openai_direct_model:
            if is_express_model_request:
                openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
            else:
                openai_handler = OpenAIDirectHandler(credential_manager=credential_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_openai_search=is_openai_search_model)
        elif is_auto_model:
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": ENCRYPTION_INSTRUCTIONS}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}
            ]
            last_err = None
            for attempt in attempts:
                current_gen_config_dict = attempt["config_modifier"](gen_config_dict.copy())
                try:
                    result = await execute_gemini_call(client_to_use, attempt["model"], attempt["prompt_func"], current_gen_config_dict, request, is_auto_attempt=True)
                    return result
                except Exception as e_auto:
                    last_err = e_auto
                    await asyncio.sleep(1)
            
            err_msg = f"All auto-mode attempts failed. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream:
                async def final_auto_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    yield f"data: {json.dumps(err_content)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_auto_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed.", "server_error"))

        else: 
            current_prompt_func = create_gemini_prompt

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]
            
            elif is_encrypted_model:
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                current_prompt_func = create_encrypted_full_gemini_prompt
            
            if not isinstance(gen_config_dict.get("thinking_config"), dict):
                gen_config_dict["thinking_config"] = {}

            if is_lite_model or is_image_model:
                gen_config_dict["thinking_config"]["include_thoughts"] = False
            else:
                gen_config_dict["thinking_config"]["include_thoughts"] = True
            
            is_pro_model = "pro" in base_model_name 
            
            if is_thinking_capable:
                if is_nothinking_model or is_max_thinking_model:
                    if is_nothinking_model:
                        budget = 128 if is_pro_model else 0
                    else:  
                        budget = 32768 if is_pro_model else 24576
                    
                    gen_config_dict["thinking_config"]["thinking_budget"] = budget
                    if budget == 0:
                        gen_config_dict["thinking_config"]["include_thoughts"] = False

            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
