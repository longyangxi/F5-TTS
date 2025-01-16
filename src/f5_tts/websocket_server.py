import argparse
import gc
import struct
import torch
import torchaudio
import traceback
import asyncio
import websockets
import json
import base64
import uuid
import os
from importlib.resources import files
from cached_path import cached_path

from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from model.backbones.dit import DiT


class TTSModel:
    """单例模式的模型类，用于共享模型实例"""
    _instance = None
    
    def __init__(self, ckpt_file, vocab_file, device=None, dtype=torch.float32):
        if TTSModel._instance is not None:
            raise Exception("This class is a singleton!")
            
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        print(f"Initializing model on device: {self.device}")
        
        # 加载模型
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)
        
        # 加载 vocoder
        self.vocoder = load_vocoder(is_local=False)
        
        # 预热模型
        self._warm_up()
        
        TTSModel._instance = self
    
    @staticmethod
    def get_instance():
        if TTSModel._instance is None:
            raise Exception("Model not initialized!")
        return TTSModel._instance
    
    def _warm_up(self):
        """只在模型第一次加载时预热一次"""
        print("Warming up the model...")
        try:
            # 使用一个简单的音频文件进行预热
            default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
            ref_audio, ref_text = preprocess_ref_audio_text(default_ref_audio, "")
            audio, sr = torchaudio.load(ref_audio)
            gen_text = "Warm-up text for the model."
            
            infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
            print("Model warm-up completed successfully.")
        except Exception as e:
            print(f"Error during warm-up: {e}")
            traceback.print_exc()
            raise


class TTSProcessor:
    """处理单个用户的 TTS 请求"""
    def __init__(self, ref_audio, ref_text):
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.sampling_rate = 24000  # 采样率
        
        # 获取共享的模型实例
        self.tts_model = TTSModel.get_instance()

    def generate_stream(self, text, play_steps_in_s=0.5):
        """Generate audio in chunks and yield them in real-time."""
        try:
            # 处理参考音频和文本
            ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
            audio, sr = torchaudio.load(ref_audio)

            # 使用共享模型进行推理
            audio_chunk, final_sample_rate, _ = infer_batch_process(
                (audio, sr),
                ref_text,
                [text],
                self.tts_model.model,
                self.tts_model.vocoder,
                device=self.tts_model.device,
            )

            # 分块发送音频
            chunk_size = int(final_sample_rate * play_steps_in_s)

            if len(audio_chunk) < chunk_size:
                packed_audio = struct.pack(f"{len(audio_chunk)}f", *audio_chunk)
                yield packed_audio
                return

            for i in range(0, len(audio_chunk), chunk_size):
                chunk = audio_chunk[i : i + chunk_size]
                if i + chunk_size >= len(audio_chunk):
                    chunk = audio_chunk[i:]
                if len(chunk) > 0:
                    packed_audio = struct.pack(f"{len(chunk)}f", *chunk)
                    yield packed_audio

        except Exception as e:
            print(f"Error generating audio stream: {e}")
            traceback.print_exc()
            raise


async def handle_websocket(websocket, processor_args):
    """Handle individual WebSocket connections"""
    print(f"New client connected")
    connection_processor = None
    temp_file_path = None
    
    try:
        async for message in websocket:
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'init':
                        # 清理之前的临时文件
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        connection_processor = None
                        
                        # 处理参考音频
                        ref_audio_base64 = data.get('ref_audio')
                        ref_text = data.get('ref_text', '')
                        
                        if not ref_audio_base64:
                            raise ValueError("Reference audio and text are required")
                        
                        try:
                            if ',' in ref_audio_base64:
                                ref_audio_base64 = ref_audio_base64.split(',')[1]
                            
                            ref_audio_data = base64.b64decode(ref_audio_base64)
                            temp_file_path = f"temp_ref_{uuid.uuid4()}.wav"
                            
                            with open(temp_file_path, 'wb') as f:
                                f.write(ref_audio_data)
                            
                            print(f"Saved reference audio to {temp_file_path}")
                            
                            # 创建新的处理器实例
                            connection_processor = TTSProcessor(temp_file_path, ref_text)
                            await websocket.send(json.dumps({'type': 'init_success'}))
                            
                        except Exception as e:
                            if temp_file_path and os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                            raise Exception(f"Failed to initialize processor: {str(e)}")
                        
                    elif message_type == 'tts':
                        if not connection_processor:
                            raise Exception("Please initialize with reference audio first")
                            
                        text = data.get('text', '').strip()
                        if text:
                            for audio_chunk in connection_processor.generate_stream(text):
                                await websocket.send(audio_chunk)
                            await websocket.send(b"END_OF_AUDIO")
                    
            except json.JSONDecodeError as e:
                print(f"Invalid JSON message: {e}")
                await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid JSON format'}))
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send(json.dumps({'type': 'error', 'message': str(e)}))
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Error removing temp file: {e}")
        print("Connection closed")


async def start_websocket_server(host, port, processor_args):
    """Start the WebSocket server"""
    # 初始化共享模型实例
    TTSModel(
        ckpt_file=processor_args['ckpt_file'],
        vocab_file=processor_args['vocab_file'],
        device=processor_args['device'],
        dtype=processor_args['dtype']
    )
    
    async with websockets.serve(
        lambda ws: handle_websocket(ws, processor_args),
        host,
        port
    ):
        print(f"WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=9998, help="Port to bind the server to")

    parser.add_argument(
        "--ckpt_file",
        default=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")),
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vocab_file",
        default="",
        help="Path to the vocab file if customized",
    )

    parser.add_argument("--device", default=None, help="Device to run the model on")
    parser.add_argument("--dtype", default=torch.float32, help="Data type to use for model inference")

    args = parser.parse_args()

    try:
        processor_args = {
            'ckpt_file': args.ckpt_file,
            'vocab_file': args.vocab_file,
            'device': args.device,
            'dtype': args.dtype,
        }

        asyncio.run(start_websocket_server(args.host, args.port, processor_args))

    except KeyboardInterrupt:
        print("\nShutting down server...")
        gc.collect()
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
