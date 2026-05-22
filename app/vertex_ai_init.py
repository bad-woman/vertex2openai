import json
from typing import Optional
from google import genai
from credentials_manager import CredentialManager, parse_multiple_json_credentials
import config as app_config
from google.genai import types
from model_loader import refresh_models_config_cache

def get_http_options(base_url: Optional[str] = None) -> Optional[types.HttpOptions]:
    """
    获取包含代理和基础URL的HttpOptions配置，采用 httpx 全版本通用的单数 proxy 参数
    """
    client_args = {}
    if app_config.PROXY_URL:
        client_args['proxy'] = app_config.PROXY_URL
        
    if base_url:
        return types.HttpOptions(
            base_url=base_url,
            client_args=client_args if client_args else None,
            async_client_args=client_args if client_args else None
        )
    elif client_args:
        return types.HttpOptions(
            client_args=client_args,
            async_client_args=client_args
        )
    return None

async def init_vertex_ai(credential_manager_instance: CredentialManager) -> bool:
    try:
        credentials_json_str = app_config.GOOGLE_CREDENTIALS_JSON_STR
        env_creds_loaded_into_manager = False

        if credentials_json_str:
            print("INFO: Found GOOGLE_CREDENTIALS_JSON environment variable. Attempting to load into CredentialManager.")
            try:
                json_objects = parse_multiple_json_credentials(credentials_json_str)
                if json_objects:
                    print(f"DEBUG: Parsed {len(json_objects)} potential credential objects from GOOGLE_CREDENTIALS_JSON.")
                    success_count = credential_manager_instance.load_credentials_from_json_list(json_objects)
                    if success_count > 0:
                        print(f"INFO: Successfully loaded {success_count} credentials from GOOGLE_CREDENTIALS_JSON into manager.")
                        env_creds_loaded_into_manager = True
                
                if not env_creds_loaded_into_manager:
                    print("DEBUG: Multi-JSON loading from GOOGLE_CREDENTIALS_JSON did not add to manager or was empty. Attempting single JSON load.")
                    try:
                        credentials_info = json.loads(credentials_json_str)
                        if isinstance(credentials_info, dict) and \
                           all(field in credentials_info for field in ["type", "project_id", "private_key_id", "private_key", "client_email"]):
                            if credential_manager_instance.add_credential_from_json(credentials_info):
                                print("INFO: Successfully loaded single credential from GOOGLE_CREDENTIALS_JSON into manager.")
                            else:
                                print("WARNING: Single JSON from GOOGLE_CREDENTIALS_JSON failed to load into manager via add_credential_from_json.")
                        else:
                             print("WARNING: Single JSON from GOOGLE_CREDENTIALS_JSON is not a valid dict or missing required fields for basic check.")
                    except json.JSONDecodeError as single_json_err:
                        print(f"WARNING: GOOGLE_CREDENTIALS_JSON could not be parsed as a single JSON object: {single_json_err}.")
                    except Exception as single_load_err:
                        print(f"WARNING: Error trying to load single JSON from GOOGLE_CREDENTIALS_JSON into manager: {single_load_err}.")
            except Exception as e_json_env:
                print(f"WARNING: Error processing GOOGLE_CREDENTIALS_JSON env var: {e_json_env}.")
        else:
            print("INFO: GOOGLE_CREDENTIALS_JSON environment variable not found.")

        print("INFO: Attempting to pre-warm model configuration cache during startup...")
        models_loaded_successfully = await refresh_models_config_cache()
        if models_loaded_successfully:
            print("INFO: Model configuration cache pre-warmed successfully.")
        else:
            print("WARNING: Failed to pre-warm model configuration cache during startup. It will be loaded lazily on first request.")

        if credential_manager_instance.refresh_credentials_list():
            total_creds = credential_manager_instance.get_total_credentials()
            print(f"INFO: Credential Manager reports {total_creds} credential(s) available (from files and/or GOOGLE_CREDENTIALS_JSON).")
            
            print("INFO: Attempting to validate a credential by creating a temporary client...")
            temp_creds_val, temp_project_id_val = credential_manager_instance.get_credentials()
            if temp_creds_val and temp_project_id_val:
                try:
                    _ = genai.Client(
                        vertexai=True, 
                        credentials=temp_creds_val, 
                        project=temp_project_id_val, 
                        location="global", 
                        http_options=get_http_options()
                    )
                    print(f"INFO: Successfully validated a credential from Credential Manager (Project: {temp_project_id_val}). Initialization check passed.")
                    return True
                except Exception as e_val:
                    print(f"WARNING: Failed to validate a random credential from manager by creating a temp client: {e_val}. App may rely on non-validated credentials.")
                    return True 
            elif total_creds > 0:
                 print(f"WARNING: {total_creds} credentials reported by manager, but could not retrieve one for validation. Problems might occur.")
                 return True 
            else:
                 print("ERROR: No credentials available after attempting to load from all sources.")
                 return False 
        else:
            print("ERROR: Credential Manager reports no available credentials after processing all sources.")
            return False

    except Exception as e:
        print(f"CRITICAL ERROR during Vertex AI credential setup: {e}")
        return False