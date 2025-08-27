import os
import sys
import shutil
import tempfile
import asyncio
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI()

from worker_router import router as worker_router
app.include_router(worker_router)

is_processing = False

def run_facefusion_process(
    source_paths: list[str],
    target_path: str, 
    output_path: str,
    processors: str,
    face_enhancer_model: str | None,
    face_enhancer_blend: int | None,
    frame_enhancer_model: str | None,
    frame_enhancer_blend: int | None,
    age_modifier_direction: int | None,
    expression_restorer_model: str | None,
    expression_restorer_factor: int | None,
    face_debugger_items: str | None,
    face_editor_model: str | None,
    face_editor_eyebrow_direction: float | None,
    face_editor_eye_gaze_horizontal: float | None,
    face_editor_eye_gaze_vertical: float | None,
    face_editor_eye_open_ratio: float | None,
    face_editor_lip_open_ratio: float | None,
    face_editor_mouth_grim: float | None,
    face_editor_mouth_pout: float | None,
    face_editor_mouth_smile: float | None,
    face_editor_mouth_position_horizontal: float | None,
    face_editor_mouth_position_vertical: float | None,
    face_editor_head_pitch: float | None,
    face_editor_head_yaw: float | None,
    face_editor_head_roll: float | None,
    frame_colorizer_model: str | None,
    frame_colorizer_blend: int | None,
    lip_syncer_model: str | None,
    lip_syncer_weight: float | None,
    deep_swapper_model: str | None,
    deep_swapper_morph: int | None,
    face_selector_mode: str | None = None,
    reference_face_distance: float | None = None,
    face_mask_types: str | None = None,
    face_mask_blur: float | None = None,
    face_mask_padding: str | None = None,
    output_video_encoder: str | None = None,
    output_video_quality: int | None = None,
    output_video_resolution: str | None = None,
    output_video_fps: float | None = None,
    face_swapper_model: str | None = None,
    face_swapper_pixel_boost : str | None = None,
    face_detector_model: str | None = None,
    face_detector_score: float | None = None,
    face_selector_order: str | None = None,
    face_selector_gender: str | None = None,
    face_selector_age_start: int | None = None,
    face_selector_age_end: int | None = None,
    output_video_preset: str | None = None
):
    global is_processing
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        run_py_path = os.path.join(project_root, 'facefusion.py')

        command = [
            sys.executable,
            run_py_path,
            'headless-run',
            '--source', *source_paths,
            '--target', target_path,
            '--output-path', output_path,
            '--execution-providers', 'cuda' # Ganti ke 'cpu' jika perlu
        ]

        processor_list = [p.strip() for p in processors.split(',')]
        command.extend(['--processors', *processor_list])
        
        if 'face_enhancer' in processor_list and face_enhancer_model:
            command.extend(['--face-enhancer-model', face_enhancer_model, '--face-enhancer-blend', str(face_enhancer_blend)])
        if 'frame_enhancer' in processor_list and frame_enhancer_model:
            command.extend(['--frame-enhancer-model', frame_enhancer_model, '--frame-enhancer-blend', str(frame_enhancer_blend)])
        if 'age_modifier' in processor_list and age_modifier_direction is not None:
            command.extend(['--age-modifier-model', 'styleganex_age', '--age-modifier-direction', str(age_modifier_direction)])
        
        if 'expression_restorer' in processor_list and expression_restorer_model:
            command.extend(['--expression-restorer-model', expression_restorer_model])
            if expression_restorer_factor is not None:
                command.extend(['--expression-restorer-factor', str(expression_restorer_factor)])

        if 'face_debugger' in processor_list and face_debugger_items:
            command.extend(['--face-debugger-items', *[item.strip() for item in face_debugger_items.split(',')]])
        if 'face_editor' in processor_list and face_editor_model:
            command.extend(['--face-editor-model', face_editor_model])
            editor_params = {
                '--face-editor-eyebrow-direction': face_editor_eyebrow_direction, '--face-editor-eye-gaze-horizontal': face_editor_eye_gaze_horizontal,
                '--face-editor-eye-gaze-vertical': face_editor_eye_gaze_vertical, '--face-editor-eye-open-ratio': face_editor_eye_open_ratio,
                '--face-editor-lip-open-ratio': face_editor_lip_open_ratio, '--face-editor-mouth-grim': face_editor_mouth_grim,
                '--face-editor-mouth-pout': face_editor_mouth_pout, '--face-editor-mouth-smile': face_editor_mouth_smile,
                '--face-editor-mouth-position-horizontal': face_editor_mouth_position_horizontal, '--face-editor-mouth-position-vertical': face_editor_mouth_position_vertical,
                '--face-editor-head-pitch': face_editor_head_pitch, '--face-editor-head-yaw': face_editor_head_yaw, '--face-editor-head-roll': face_editor_head_roll
            }
            for key, value in editor_params.items():
                if value is not None: command.extend([key, str(value)])
        if 'frame_colorizer' in processor_list and frame_colorizer_model:
            command.extend(['--frame-colorizer-model', frame_colorizer_model, '--frame-colorizer-blend', str(frame_colorizer_blend)])
        if 'lip_syncer' in processor_list and lip_syncer_model:
            command.extend(['--lip-syncer-model', lip_syncer_model, '--lip-syncer-weight', str(lip_syncer_weight)])
        if 'deep_swapper' in processor_list and deep_swapper_model:
            command.extend(['--deep-swapper-model', deep_swapper_model, '--deep-swapper-morph', str(deep_swapper_morph)])
        if face_selector_mode:
            command.extend(['--face-selector-mode', face_selector_mode])
            if face_selector_mode == 'reference':
                command.extend(['--reference-face-distance', str(reference_face_distance)])
        if face_mask_types:
            command.extend(['--face-parser-model', 'bisenet_resnet_18'])
            mask_types_list = [item.strip() for item in face_mask_types.split(',')]
            command.extend(['--face-mask-types', *mask_types_list])
            if face_mask_blur is not None:
                command.extend(['--face-mask-blur', str(face_mask_blur)])
            if face_mask_padding is not None:
                padding_list = [item.strip() for item in face_mask_padding.split(',')]
                command.extend(['--face-mask-padding', *padding_list])
        if output_video_encoder:
            command.extend(['--output-video-encoder', output_video_encoder])
        if output_video_quality is not None:
            command.extend(['--output-video-quality', str(output_video_quality)])
        if output_video_resolution:
            command.extend(['--output-video-resolution', output_video_resolution])
        if output_video_fps is not None:
            command.extend(['--output-video-fps', str(output_video_fps)])
        if 'face_swapper' in processor_list:
            if face_swapper_model:
                command.extend(['--face-swapper-model', face_swapper_model])
            if face_swapper_pixel_boost is not None:
                command.extend(['--face-swapper-pixel-boost', str(face_swapper_pixel_boost)])
        if face_detector_model:
            command.extend(['--face-detector-model', face_detector_model])
        if face_detector_score is not None:
            command.extend(['--face-detector-score', str(face_detector_score)])
        if face_selector_order:
            command.extend(['--face-selector-order', face_selector_order])
        if face_selector_gender:
            command.extend(['--face-selector-gender', face_selector_gender])
        if face_selector_age_start is not None:
            command.extend(['--face-selector-age-start', str(face_selector_age_start)])
        if face_selector_age_end is not None:
            command.extend(['--face-selector-age-end', str(face_selector_age_end)])
        if output_video_preset:
            command.extend(['--output-video-preset', output_video_preset])

        print(f"Menjalankan perintah: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=project_root)
        print("--- PROSES BERHASIL ---", result.stdout)

    except subprocess.CalledProcessError as e:
        print("--- PROSES GAGAL ---")
        print("Return Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
    finally:
        is_processing = False

@app.post("/swap/")
def create_swap(
    # Kebutuhan untuk upload file
    source_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    audio_file: UploadFile = File(None),
    # Pilihan fitur atau processor nya
    processors: str = Form("face_swapper"),
    #Processors
    face_enhancer_model: str = Form(None),
    face_enhancer_blend: int = Form(50),
    frame_enhancer_model: str = Form(None),
    frame_enhancer_blend: int = Form(80),
    age_modifier_direction: int = Form(None),
    expression_restorer_model: str = Form("live_portrait"),
    expression_restorer_factor: int = Form(None), 
    face_debugger_items: str = Form(None),
    face_editor_model: str = Form('live_portrait'),
    face_editor_eyebrow_direction: float = Form(None),
    face_editor_eye_gaze_horizontal: float = Form(None),
    face_editor_eye_gaze_vertical: float = Form(None),
    face_editor_eye_open_ratio: float = Form(None),
    face_editor_lip_open_ratio: float = Form(None),
    face_editor_mouth_grim: float = Form(None),
    face_editor_mouth_pout: float = Form(None),
    face_editor_mouth_smile: float = Form(None),
    face_editor_mouth_position_horizontal: float = Form(None),
    face_editor_mouth_position_vertical: float = Form(None),
    face_editor_head_pitch: float = Form(None),
    face_editor_head_yaw: float = Form(None),
    face_editor_head_roll: float = Form(None),
    frame_colorizer_model: str = Form(None),
    frame_colorizer_blend: int = Form(80),
    lip_syncer_model: str = Form(None),
    lip_syncer_weight: float = Form(1.0),
    deep_swapper_model: str = Form(None),
    deep_swapper_morph: int = Form(80),
    # Parameter tambahan atau opsional (Bukan Processor)
    face_selector_mode: str = Form(None),
    reference_face_distance: float = Form(0.3),
    face_mask_types: str = Form(None),
    face_mask_blur: float = Form(0.3),
    face_mask_padding: str = Form(None),
    output_video_encoder: str = Form("libx264"),
    output_video_quality: int = Form(80),
    output_video_resolution: str = Form(None),
    output_video_fps: float = Form(None),
    face_swapper_model: str = Form(None),
    face_swapper_pixel_boost: str = Form(None),
    face_detector_model: str = Form(None),
    face_detector_score: float = Form(None),
    face_selector_order: str = Form(None),
    face_selector_gender: str = Form(None),
    face_selector_age_start: int = Form(None),
    face_selector_age_end: int = Form(None),
    output_video_preset: str = Form('medium')
):
    
    global is_processing
    if is_processing:
        raise HTTPException(status_code=429, detail="Server sedang memproses permintaan lain.")
    
    is_processing = True
    temp_dir = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(temp_dir)
        print(f"Direktori temporary {temp_dir} telah dihapus.")

    try:
        source_extension = os.path.splitext(source_file.filename)[1]
        source_path = os.path.join(temp_dir, f"source{source_extension}")
        source_paths = [source_path]

        target_extension = os.path.splitext(target_file.filename)[1]
        target_path = os.path.join(temp_dir, f"target{target_extension}")
        output_filename = f"result_output{target_extension}"
        output_path = os.path.join(temp_dir, output_filename)

        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_file.file, f)
        
        if audio_file:
            audio_extension = os.path.splitext(audio_file.filename)[1]
            audio_path = os.path.join(temp_dir, f"audio{audio_extension}")
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            source_paths.append(audio_path)

        print("File berhasil di-upload. Memulai proses FaceFusion (mode blocking)...")

        run_facefusion_process(
            source_paths, target_path, output_path,
            processors, face_enhancer_model, face_enhancer_blend,
            frame_enhancer_model, frame_enhancer_blend,
            age_modifier_direction,
            expression_restorer_model, expression_restorer_factor,
            face_debugger_items,
            face_editor_model,
            face_editor_eyebrow_direction, face_editor_eye_gaze_horizontal, face_editor_eye_gaze_vertical,
            face_editor_eye_open_ratio, face_editor_lip_open_ratio, face_editor_mouth_grim,
            face_editor_mouth_pout, face_editor_mouth_smile, face_editor_mouth_position_horizontal,
            face_editor_mouth_position_vertical, face_editor_head_pitch, face_editor_head_yaw, face_editor_head_roll,
            frame_colorizer_model, frame_colorizer_blend,
            lip_syncer_model, lip_syncer_weight,
            deep_swapper_model, deep_swapper_morph,
            face_selector_mode, reference_face_distance,
            face_mask_types, face_mask_blur, face_mask_padding,
            output_video_encoder, output_video_quality,
            output_video_resolution, output_video_fps,
            face_swapper_model, face_swapper_pixel_boost,
            face_detector_model, face_detector_score,
            face_selector_order, face_selector_gender, face_selector_age_start, face_selector_age_end,
            output_video_preset
        )

        if not os.path.exists(output_path):
            is_processing = False
            raise HTTPException(status_code=500, detail="Pemrosesan gagal. File output tidak ditemukan.")
            
        print(f"Proses selesai. Mengirim file hasil: {output_path}")

        return FileResponse(
            path=output_path,
            media_type='video/mp4',
            filename=output_filename,
            background=BackgroundTask(cleanup)
        )
    except Exception as e:
        cleanup()
        is_processing = False
        error_detail = f"Terjadi error internal: {str(e)}"
        if isinstance(e, subprocess.CalledProcessError):
            error_detail = f"Proses FaceFusion gagal. STDERR: {e.stderr}"
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8081)