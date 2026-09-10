"""Launch with: python -m sam3_vlm.gui.app (install the optional [gui] extra)."""

import argparse
import logging
from pathlib import Path

from sam3_vlm.experiments.m8_smoke import load_m8_config
from sam3_vlm.gui.service import SingleImageService
from sam3_vlm.gui.settings import CONTROLS, CONTROL_GROUPS, build_config, control_values, presets


def _form_values(config):
    # Some Gradio versions render Number(None) as zero. Optional numeric fields
    # must distinguish blank from zero (especially GT and hard-count cutoffs).
    return ["" if value is None else str(value) if control.nullable else value
            for control, value in zip(CONTROLS, control_values(config))]


def create_app(deployment, service=None):
    import gradio as gr

    service = service or SingleImageService(deployment)
    variants = presets(deployment.v4_config)
    default = variants["D_Qwen_TwoRound"]
    controls = []
    with gr.Blocks(title="SAM3 + Qwen · V4", analytics_enabled=False) as app:
        gr.Markdown("# SAM3 + Qwen\nUpload an image, describe the fruit, and adjust the settings. "
                    "Qwen's current instructions and tree-fruit scope stay fixed.")
        with gr.Row():
            preset = gr.Dropdown(list(variants), value=default.name, label="Experiment preset")
            apply_preset = gr.Button("Apply preset")
        gr.Markdown("Presets fill the controls below; your edits are used when you click **Run**. "
                    "Qwen off reports candidate count (A/B). Qwen on defaults to soft count. "
                    "The call limit is a maximum; the controller can stop earlier.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", sources=["upload"], label="Input image")
                prompt = gr.Textbox(value="green fruit", label="Target description")
                enable_qwen = gr.Checkbox(value=True, label="Enable Qwen")
                with gr.Row():
                    gt = gr.Textbox(value="", label="GT count (optional)", placeholder="Leave blank if unknown")
                    seed = gr.Number(value=deployment.seed, minimum=0, label="Random seed")
            with gr.Column():
                output_image = gr.Image(type="filepath", label="Final candidate boxes", interactive=False)
                status = gr.Markdown("Ready. Models load on the first run.")
                with gr.Accordion("Detailed results and errors", open=False):
                    result = gr.JSON(label="Count, calls and errors")
                files = gr.File(label="Download results", file_count="multiple", interactive=False)
        gr.Markdown("Images show every active candidate as a plain rectangle, with no labels or scores. "
                    "Their number can differ from a soft count.")
        run = gr.Button("Run", variant="primary")
        defaults = iter(_form_values(default.config))
        for index, (group_name, group) in enumerate(CONTROL_GROUPS.items()):
            with gr.Accordion(group_name, open=index == 0):
                for offset in range(0, len(group), 2):
                    with gr.Row():
                        for control in group[offset:offset + 2]:
                            value = next(defaults)
                            kwargs = dict(value=value, label=control.label, info=control.info or None)
                            if control.kind == "bool":
                                component = gr.Checkbox(**kwargs)
                            elif control.kind == "text" or control.nullable:
                                component = gr.Textbox(**kwargs)
                            else:
                                # Do not round integer inputs: server validation rejects fractions.
                                component = gr.Number(**kwargs, minimum=control.minimum,
                                                      maximum=control.maximum,
                                                      step=1 if control.kind == "int" else .01)
                            controls.append(component)
        with gr.Accordion("Effective settings from the last successful run", open=False):
            resolved = gr.JSON(label="Recorded configuration")

        def load_preset(name):
            variant = variants[name]
            return [variant.uses_qwen, *_form_values(variant.config)]

        def execute(image, prompt, enabled, gt, seed, *values):
            try:
                config = build_config(default.config, enabled, list(values))
                row, request, overlay, downloads = service.run(image, prompt, config, enabled, gt, seed)
                return (overlay, f"Completed · count **{row['predicted_count']:.3f}** · "
                        f"{row['qwen_calls']} Qwen calls · {row['runtime_seconds']:.2f} s", row, downloads, request)
            except Exception as exc:
                logging.getLogger(__name__).exception("GUI run failed")
                # Clear previous outputs so an unsuccessful request cannot display stale detections.
                return None, "Run failed. Open **Detailed results and errors** for the reason.", {"success": False, "error": str(exc)}, [], None

        apply_preset.click(load_preset, inputs=[preset], outputs=[enable_qwen, *controls], queue=False)
        run.click(execute, inputs=[image, prompt, enable_qwen, gt, seed, *controls],
                  outputs=[output_image, status, result, files, resolved], concurrency_limit=1)
    return app.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m8_real_smoke.json")
    parser.add_argument("--output-dir", dest="output_dir", default="runs/gui")
    parser.add_argument("--sam3-model", default=None)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error(f"Config not found: {args.config}. Run from the V4 repository or pass --config.")
    deployment = load_m8_config(args, config_path=args.config)
    create_app(deployment).launch(server_name=args.host, server_port=args.port, share=False,
                                  allowed_paths=[str(Path(deployment.output_root).resolve())])


if __name__ == "__main__":
    main()
