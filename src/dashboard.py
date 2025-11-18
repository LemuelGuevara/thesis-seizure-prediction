from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import torch
import torch.nn.functional as F

# your model & helpers
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.preprocessing.check_mosaic import load_npz_data, plot_spectrogram, plot_bispectrum

def _path_from_gradio_file(f):
    """Return filesystem path for a Gradio upload object (or string)."""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if hasattr(f, "name"):
        return f.name
    if isinstance(f, dict):
        # try common keys (Gradio older/newer variants)
        for key in ("name", "tmp_path", "tempfile", "file"):
            if key in f:
                return f[key]
        if "data" in f:
            tmp = Path("/tmp") / f.get("name", "upload.tmp")
            with open(tmp, "wb") as fh:
                fh.write(f["data"])
            return str(tmp)
    return None

def on_tf_upload(file_obj):
    if file_obj is None:
        return "No file selected.", None
    path = _path_from_gradio_file(file_obj)
    if path is None:
        return "Could not determine uploaded file path.", None
    filename = Path(path).name
    if not filename.lower().endswith(".npz"):
        return f"Unsupported file type: {filename}", None

    try:
        tensor, start, end, phase, freqs, n_channels = load_npz_data(path)
    except Exception as e:
        return f"Failed loading '{filename}': {e}", None

    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = np.moveaxis(tensor, 0, -1)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        return f"Expected RGB spectrogram (H,W,3); got shape {tensor.shape}", None

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        plot_spectrogram(ax, tensor, start, end, phase, freqs, n_channels)
        plt.tight_layout()
        return filename, fig
    except Exception as e:
        plt.close(fig)
        return f"Error plotting '{filename}': {e}", None

def on_bis_upload(file_obj):
    if file_obj is None:
        return "No file selected.", None
    path = _path_from_gradio_file(file_obj)
    if path is None:
        return "Could not determine uploaded file path.", None
    filename = Path(path).name
    if not filename.lower().endswith(".npz"):
        return f"Unsupported file type: {filename}", None

    try:
        tensor, start, end, phase, freqs, n_channels = load_npz_data(path)
    except Exception as e:
        return f"Failed loading '{filename}': {e}", None

    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = np.moveaxis(tensor, 0, -1)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        return f"Expected RGB bispectrum (H,W,3); got shape {tensor.shape}", None

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        plot_bispectrum(ax, tensor, phase, start, end, freqs)
        plt.tight_layout()
        return filename, fig
    except Exception as e:
        plt.close(fig)
        return f"Error plotting '{filename}': {e}", None

def load_model_from_upload(model_file_obj, device=torch.device("cpu")):
    """
    Load a .pt containing a state_dict and return an nn.Module ready for inference.
    Uses the exact pattern you requested:
      model = TheModelClass(*args, **kwargs)
      model.load_state_dict(torch.load(PATH, weights_only=True))
      model.eval()
    """
    if model_file_obj is None:
        return None, "No model file uploaded."
    path = _path_from_gradio_file(model_file_obj)
    if path is None:
        return None, "Could not determine model file path."
    filename = Path(path).name
    if not filename.lower().endswith(".pt"):
        return None, f"Unsupported model type: {filename} (expect .pt)"

    try:
        # instantiate model (modify constructor args if needed)
        model = MultimodalSeizureModel(use_cbam=True)

        # load only weights (state_dict); user-requested syntax included
        state = torch.load(path, map_location=device, weights_only=True)

        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, f"Loaded model state_dict from '{filename}'."
    except Exception as e:
        return None, f"Failed to load state_dict from '{filename}': {e}"

def preprocess_for_model(img_np, expected_size=(224, 224), device=torch.device("cpu")):
    """
    img_np: numpy array (H,W,3) or (3,H,W)
    Returns torch tensor (1,3,H,W) on given device (float32).
    """
    img = img_np.astype("float32")
    mean = img.mean()
    std = img.std() if img.std() > 0 else 1.0
    img = (img - mean) / std
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    if (t.shape[2], t.shape[3]) != expected_size:
        t = F.interpolate(t, size=expected_size, mode="bilinear", align_corners=False)
    return t.to(device)

def predict_from_uploaded(tf_file_obj, bis_file_obj, model_obj, device=torch.device("cpu"), expected_size=(224,224)):
    """
    model_obj must be the nn.Module returned by load_model_from_upload (Gradio state).
    Returns (label_str, confidence_str)
    """
    # consistent return type: (label: str, confidence: str)
    if model_obj is None:
        return "No model loaded.", "0.00%"

    tf_path = _path_from_gradio_file(tf_file_obj)
    bis_path = _path_from_gradio_file(bis_file_obj)
    if tf_path is None or bis_path is None:
        return "Please upload both TF and Bispectrum .npz files.", "0.00%"

    try:
        tf_tensor_np, _, _, _, _, _ = load_npz_data(tf_path)
        bis_tensor_np, _, _, _, _, _ = load_npz_data(bis_path)
    except Exception as e:
        return f"Error loading npz files: {e}", "0.00%"

    if tf_tensor_np.ndim == 3 and tf_tensor_np.shape[0] in (1,3):
        tf_tensor_np = np.moveaxis(tf_tensor_np, 0, -1)
    if bis_tensor_np.ndim == 3 and bis_tensor_np.shape[0] in (1,3):
        bis_tensor_np = np.moveaxis(bis_tensor_np, 0, -1)

    if tf_tensor_np.ndim != 3 or tf_tensor_np.shape[2] != 3:
        return f"TF file tensor shape not RGB (H,W,3): {tf_tensor_np.shape}", "0.00%"
    if bis_tensor_np.ndim != 3 or bis_tensor_np.shape[2] != 3:
        return f"Bis file tensor shape not RGB (H,W,3): {bis_tensor_np.shape}", "0.00%"

    try:
        tf_t = preprocess_for_model(tf_tensor_np, expected_size=expected_size, device=device)
        bis_t = preprocess_for_model(bis_tensor_np, expected_size=expected_size, device=device)
    except Exception as e:
        return f"Preprocessing error: {e}", "0.00%"

    model = model_obj  # already a nn.Module (from load_model_from_upload)
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            out = model(tf_t, bis_t)  
        except TypeError:
            try:
                cat = torch.cat([tf_t, bis_t], dim=1)
                out = model(cat)
            except Exception as e:
                return f"Model forward failed (dual-input and concat tried): {e}", "0.00%"
        except Exception as e:
            return f"Model forward error: {e}", "0.00%"

    try:
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.detach().cpu()
        if logits.numel() == 1:
            score = float(logits.item())
            label = "Seizure" if score > 0.5 else "No seizure"
            conf_pct = score * 100.0 if score > 0.5 else (1.0 - score) * 100.0
            return label, f"{conf_pct:.2f}%"
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        label_idx = int(idx.item())
        confidence = float(conf.item())
        label_map = {0: "Interictal (Non-seizure)", 1: "Preictal (Seizure)"}
        label = label_map.get(label_idx, f"class_{label_idx}")
        return label, f"{confidence * 100.0:.2f}%"
    except Exception as e:
        return f"Postprocessing error: {e}", "0.00%"

def main():
    with gr.Blocks(css="""
        body, .gradio-container, #root {
        background-color: #fbfcf8 !important;
        }
        .gr-column, .gr-group {
        background: #f9f7fa !important;
        }
        .title-center, .authors-center, .model-note-center, .input-note-center {
            text-align: center !important;
            width: 100% !important;
            display: flex;
            justify-content: center;
        }
        .title-center .gr-markdown,
        .authors-center .gr-markdown, 
        .model-note-center .gr-markdown,
        .input-note-center .gr-markdown {
            text-align: center !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .inputs-row {
            display: flex;
            justify-content: center;    
            gap: 0.5px;                  
            align-items: flex-start;
            width: 100%;
            box-sizing: border-box;
            margin-left: 0;
            margin-right: 0;
        }

        .input-col {
            width: 430px;             
            padding: 10px 14px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }

        button, .gr-button {
            border: 2px solid #525452 !important;
            box-shadow: none !important;
        }

        input[type="file"] + button, .gr-file-upload {
            border-radius: 8px !important;
        }

        .gr-plot {
            max-width: 100%;
        }

        .predict-col {
            display:flex;
            justify-content:center;
            align-items:center;
        }
    """) as demo:

        with gr.Row(elem_classes="title-center"):
            gr.Markdown("# Seizure Prediction Using EfficientNetB0+CBAM With Gated Fusion")

        with gr.Row(elem_classes="authors-center"):
            gr.Markdown("## Dela Vega | Facturan | Guevera | Villalobos")

        with gr.Row(elem_classes="model-note-center"):
            gr.Markdown("#### *Note: The model expects that the uploaded test cases are of the same patient used during training.*")

        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=2):
                model_upload = gr.UploadButton(label="Upload model (.pt)", file_types=[".pt"])
                model_status = gr.Textbox(label="Model status", interactive=False)
                model_state = gr.State(value=None)
                model_upload.upload(load_model_from_upload, inputs=model_upload, outputs=[model_state, model_status])
            gr.Column(scale=1)

        with gr.Row(elem_classes="input-note-center"):
            gr.Markdown("#### *Note: Uploaded time-frequency and bispectrum files must be of the same patient and same epoch.*")

        with gr.Group("Input Files"):
            with gr.Row(elem_classes="inputs-row"):
                with gr.Column(elem_classes="input-col"):
                    gr.Markdown("### Input Time-Frequency File (.npz)")
                    tf_input = gr.UploadButton(label="Upload Time-Frequency File", file_types=[".npz"])
                    tf_filename = gr.Textbox(label="Selected Time-Frequency File", interactive=False)
                    tf_plot = gr.Plot(label="Spectrogram Preview")
                    tf_input.upload(on_tf_upload, inputs=tf_input, outputs=[tf_filename, tf_plot])
                with gr.Column(elem_classes="input-col"):
                    gr.Markdown("### Input Bispectrum File (.npz)")
                    bis_input = gr.UploadButton(label="Upload Bispectrum File", file_types=[".npz"])
                    bis_filename = gr.Textbox(label="Selected Bispectrum File", interactive=False)
                    bis_plot = gr.Plot(label="Bispectrum Preview")
                    bis_input.upload(on_bis_upload, inputs=bis_input, outputs=[bis_filename, bis_plot])

        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=1, elem_classes="predict-col"):
                predict_button = gr.Button("Predict")
            gr.Column(scale=1)

        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=2):
                predicted_label = gr.Label(label="Predicted Label")
                predicted_confidence = gr.Label(label="Confidence")
            gr.Column(scale=1)

        predict_button.click(
            fn=predict_from_uploaded,
            inputs=[tf_input, bis_input, model_state],
            outputs=[predicted_label, predicted_confidence],
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()