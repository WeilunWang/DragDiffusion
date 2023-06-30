import os
import datetime
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer

parser = ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
args = parser.parse_args()

cache_dir = args.cache_dir

device = 'cuda'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(upload_image, global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    init_image = Image.open(upload_image.name)
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(np.array(init_image))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(np.array(image_draw))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state

valid_checkpoints_dict = {
    'runwayml/stable-diffusion-v1-5': 512,
    'stabilityai/stable-diffusion-2-base': 512,
    'stabilityai/stable-diffusion-2': 768,
    'stabilityai/stable-diffusion-2-1': 768,
}

init_pkl = 'runwayml/stable-diffusion-v1-5'

with gr.Blocks() as app:

    # renderer = Renderer()
    global_state = gr.State({
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        'mask':
        None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "prompt": "A photo of a racoon",
            "motion_lambda": 0.1,
            "r1_in_pixels": 1,
            "r2_in_pixels": 3,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl
    })

    # init image
    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state
    state['renderer'].init_network(
        state['generator_params'],  # res
        state['pretrained_weight'],  # pkl
        state['params']['seed'],  # w0_seed,
        None,  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    with gr.Row():

        with gr.Row():

            # Left --> tools
            with gr.Column(scale=3):

                # Pickle
                with gr.Row():

                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Pickle', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(valid_checkpoints_dict.keys()),
                            label="Pretrained Model",
                            value=init_pkl,
                        )

                # Latent
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Latent', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_prompt = gr.Textbox(
                            value=global_state.value['params']['prompt'],
                            interactive=True,
                            label="Prompt",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="Step Size")

                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_upload_image = gr.UploadButton("Upload Image", file_types=["image"])
                            with gr.Column(scale=1, min_width=10):
                                form_finetune_model = gr.Button("Finetune Model")

                # Drag
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Drag', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button('Add Points')
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button('Reset Points')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(value=0,
                                                      label="Steps",
                                                      interactive=False)

                # Mask
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Mask', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        enable_add_mask = gr.Button('Edit Flexible Area')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]
                                ["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False)

            # Right --> Image
            with gr.Column(scale=8):
                form_image = ImageMask(
                    brush_radius=20).style(
                        width=768,
                        height=768)  # NOTE: hard image size code here.

    gr.Markdown("""
        ## Quick Start

        1. Select desired `Pretrained Model` and adjust `Seed` to generate an
           initial image.
        2. Click on image to add control points.
        3. Click `Start` and enjoy it!

        ## Advance Usage

        1. Change `Step Size` to adjust learning rate in drag optimization.
        2. Select `w` or `w+` to change latent space to optimize:
        * Optimize on `w` space may cause greater influence to the image.
        * Optimize on `w+` space may work slower than `w`, but usually achieve
          better results.
        * Note that changing the latent space will reset the image, points and
          mask (this has the same effect as `Reset Image` button).
        3. Click `Edit Flexible Area` to create a mask and constrain the
           unmasked region to remain unchanged.
        """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """)

    # Network & latents tab listeners
    def on_change_pretrained_dropdown(pretrained_value, global_state):
        """Function to handle model change.
        1. Set pretrained value to global_state
        2. Re-init images and clear all states
        """

        if isinstance(global_state, gr.State):
            state = global_state.value
        else:
            state = global_state
        global_state['pretrained_weight'] = pretrained_value
        state['renderer'].init_network(
            state['generator_params'],  # res
            state['pretrained_weight'],  # pkl
            state['params']['seed'],  # w0_seed,
            None,  # w_load
            state['params']['latent_space'] == 'w+',  # w_plus
            'const',
            state['params']['trunc_psi'],  # trunc_psi,
            state['params']['trunc_cutoff'],  # trunc_cutoff,
            None,  # input_transform
            state['params']['lr']  # lr,
        )
        clear_state(global_state)

        return global_state, global_state["images"]['image_show']

    form_pretrained_dropdown.change(
        on_change_pretrained_dropdown,
        inputs=[form_pretrained_dropdown, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_upload_image(upload_file, global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(upload_file, global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_upload_image.upload(
        on_click_upload_image,
        inputs=[form_upload_image, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_finetune_model(global_state):
        """Finetune model using LoRA"""
        logdir = f"tmpdata/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        # logdir = "tmpdata/testdata"
        os.makedirs(os.path.join(logdir, "images"), exist_ok=True)
        global_state["images"]["image_orig"].save(os.path.join(logdir, "images", "img.png"))

        yield (
            global_state,
            # gr.File.update(visible=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),

            # update other comps
            gr.Dropdown.update(interactive=False),
            gr.Textbox.update(interactive=False),
            gr.Number.update(interactive=False),
            gr.Checkbox.update(interactive=False),
            gr.Number.update(interactive=False),
        )

        os.system(f"CUDA_VISIBLE_DEVICES=2 /data3/wangwl/anaconda3/envs/stablediffusion/bin/python \
                      viz/train_dreambooth_lora.py \
                      --pretrained_model_name_or_path={global_state['pretrained_weight']}  \
                      --instance_data_dir={os.path.join(logdir, 'images')} \
                      --output_dir={os.path.join(logdir, 'weights')} \
                      --instance_prompt=\"{global_state['params']['prompt']}\" \
                      --resolution=512 \
                      --train_batch_size=1 \
                      --gradient_accumulation_steps=1 \
                      --checkpointing_steps=200 \
                      --learning_rate=2e-4 \
                      --lr_scheduler='constant' \
                      --lr_warmup_steps=0 \
                      --max_train_steps=200 \
                      --mixed_precision='fp16'")

        renderer = global_state['renderer']
        renderer.G.unet.load_attn_procs(os.path.join(logdir, "weights"))

        yield (
            global_state,
            # gr.File.update(visible=True, value=fp.name),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),

            # update other comps
            gr.Dropdown.update(interactive=True),
            gr.Textbox.update(interactive=True),
            gr.Number.update(interactive=True),
            gr.Checkbox.update(interactive=True),
            gr.Number.update(interactive=True),
        )

    form_finetune_model.click(
        on_click_finetune_model,
        inputs=[global_state],
        outputs=[
            global_state,
            # form_download_result_file,
            # >>> buttons
            form_finetune_model,
            enable_add_points,
            enable_add_mask,
            undo_points,
            form_reset_mask_btn,
            form_start_btn,
            form_stop_btn,
            # <<< buttonm
            # >>> inputs comps
            form_pretrained_dropdown,
            form_prompt,
            form_lr_number,
            show_mask,
            form_lambda_number,
        ],
    )

    # Update parameters
    def on_change_update_prompt(prompt, global_state):
        """Function to handle generation seed change.
        1. Set seed to global_state
        2. Re-init images and clear all states
        """

        global_state["params"]["prompt"] = prompt

        return global_state

    form_prompt.change(
        on_change_update_prompt,
        inputs=[form_prompt, global_state],
        outputs=[global_state],
    )

    # ==== Params
    form_lambda_number.change(
        partial(on_change_single_global_state, ["params", "motion_lambda"]),
        inputs=[form_lambda_number, global_state],
        outputs=[global_state],
    )

    def on_change_lr(lr, global_state):
        if lr == 0:
            print('lr is 0, do nothing.')
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)
        return global_state

    form_lr_number.change(
        on_change_lr,
        inputs=[form_lr_number, global_state],
        outputs=[global_state],
    )

    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # handle of start drag in mask editing mode
        global_state = preprocess_mask_info(global_state, image)

        # Prepare the points for the inference
        if len(global_state["points"]) == 0:
            # yield on_click_start_wo_points(global_state, image)
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (
                global_state,
                0,
                global_state['images']['image_show'],
                # gr.File.update(visible=False),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button
                gr.Button.update(interactive=False),

                # update other comps
                gr.Dropdown.update(interactive=True),
                gr.Textbox.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
            )
        else:
            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state['mask']).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]

            # DDIM inversion
            ddim_latent = None
            for ddim_latent, ddim_image in renderer._ddim_inversion(global_state["images"]["image_raw"],
                                                                    global_state["params"]["prompt"]):
                global_state['images']['image_show'] = ddim_image
                yield (
                    global_state,
                    0,
                    global_state['images']['image_show'],
                    # gr.File.update(visible=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),

                    # update other comps
                    gr.Dropdown.update(interactive=False),
                    gr.Textbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                )
            ddim_latent = ddim_latent.detach().clone()
            renderer.set_z_and_z_optim(ddim_latent, global_state['params']['lr'])

            global_state['temporal_params']['stop'] = False
            global_state['editing_state'] = 'running'

            # reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            step_idx = 0
            text_input = renderer.G.tokenizer(
                [global_state["params"]["prompt"]],
                padding="max_length",
                max_length=renderer.G.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_embeds = renderer.G.text_encoder(text_input.input_ids.to(device))[0]
            renderer.G.scheduler.set_timesteps(50, device=device)
            timestep = renderer.G.scheduler.timesteps[-40]
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                # do drage here!
                renderer._render_drag_impl(
                    global_state['generator_params'],
                    timestep,
                    prompt_embeds,
                    p_to_opt,  # point
                    t_to_opt,  # target
                    drag_mask,  # mask,
                    global_state['params']['motion_lambda'],  # lambda_mask
                    # reg=0,
                    # feature_idx=5,  # NOTE: do not support change for now
                    r1=global_state['params']['r1_in_pixels'],  # r1
                    r2=global_state['params']['r2_in_pixels'],  # r2
                    # random_seed     = 0,
                    # noise_mode      = 'const',
                    # trunc_psi=global_state['params']['trunc_psi'],
                    # force_fp32      = False,
                    # layer_name      = None,
                    # sel_channels    = 3,
                    # base_channel    = 0,
                    # img_scale_db    = 0,
                    # img_normalize   = False,
                    # untransform     = False,
                    is_drag=True,
                    to_pil=True)

                if step_idx % global_state['draw_interval'] == 0:
                    print('Current Source:')
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                   t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1],
                            p_i[0],
                        ]
                        global_state["points"][key_point]["target"] = [
                            t_i[1],
                            t_i[0],
                        ]
                        start_temp = global_state["points"][key_point][
                            "start_temp"]
                        print(f'    {start_temp}')

                    with torch.no_grad():
                        image_result = renderer.G.decode_latents(renderer.z.detach())[0]
                    image_result = Image.fromarray((image_result * 255).astype(np.uint8))
                    image_draw = update_image_draw(
                        image_result,
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state,
                    )
                    global_state['images']['image_raw'] = image_result

                yield (
                    global_state,
                    step_idx,
                    global_state['images']['image_show'],
                    # gr.File.update(visible=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=True),

                    # update other comps
                    gr.Dropdown.update(interactive=False),
                    gr.Textbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                )

                # increate step
                step_idx += 1

            ddim_image = None
            ddim_latent = renderer.z.detach().clone()
            for ddim_latent, ddim_image in renderer._ddim_inference(ddim_latent, global_state["params"]["prompt"]):
                global_state['images']['image_show'] = ddim_image
                yield (
                    global_state,
                    step_idx,
                    global_state['images']['image_show'],
                    # gr.File.update(visible=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=True),

                    # update other comps
                    gr.Dropdown.update(interactive=False),
                    gr.Textbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                )
            image_result = Image.fromarray((ddim_image * 255).astype(np.uint8))
            global_state['images']['image_raw'] = image_result
            image_draw = update_image_draw(image_result,
                                           global_state['points'],
                                           global_state['mask'],
                                           global_state['show_mask'],
                                           global_state)

            # fp = NamedTemporaryFile(suffix=".png", delete=False)
            # image_result.save(fp, "PNG")

            global_state['editing_state'] = 'add_points'

            yield (
                global_state,
                0,  # reset step to 0 after stop.
                global_state['images']['image_show'],
                # gr.File.update(visible=True, value=fp.name),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button with loop finish
                gr.Button.update(interactive=False),

                # update other comps
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_steps_number,
            form_image,
            # form_download_result_file,
            # >>> buttons
            form_finetune_model,
            enable_add_points,
            enable_add_mask,
            undo_points,
            form_reset_mask_btn,
            form_start_btn,
            form_stop_btn,
            # <<< buttonm
            # >>> inputs comps
            form_pretrained_dropdown,
            form_prompt,
            form_lr_number,
            show_mask,
            form_lambda_number,
        ],
    )

    def on_click_stop(global_state):
        """Function to handle stop button is clicked.
        1. send a stop signal by set global_state["temporal_params"]["stop"] as True
        2. Disable Stop button
        """
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state, form_stop_btn])

    form_draw_interval_number.change(
        partial(
            on_change_single_global_state,
            "draw_interval",
            map_transform=lambda x: int(x),
        ),
        inputs=[form_draw_interval_number, global_state],
        outputs=[global_state],
    )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    # Mask
    def on_click_reset_mask(global_state):
        global_state['mask'] = np.ones(
            (
                global_state["images"]["image_raw"].size[1],
                global_state["images"]["image_raw"].size[0],
            ),
            dtype=np.uint8,
        )
        image_draw = update_image_draw(global_state['images']['image_raw'],
                                       global_state['points'],
                                       global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Image
    def on_click_enable_draw(global_state, image):
        """Function to start add mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to add_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_click_remove_draw(global_state, image):
        """Function to start remove mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to remove_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['edinting_state'] = 'remove_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    enable_add_mask.click(on_click_enable_draw,
                          inputs=[global_state, form_image],
                          outputs=[
                              global_state,
                              form_image,
                          ])

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """

        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'], mask,
                                       global_state['show_mask'], global_state)

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    enable_add_points.click(on_click_add_point,
                            inputs=[global_state, form_image],
                            outputs=[global_state, form_image])

    def on_click_image(global_state, evt: gr.SelectData):
        """This function only support click for point selection
        """
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                  'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_clear_points(global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        clear_state(global_state, target='point')

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, {}, global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    undo_points.click(on_click_clear_points,
                      inputs=[global_state],
                      outputs=[global_state, form_image])

    def on_click_show_mask(global_state, show_mask):
        """Function to control whether show mask on image."""
        global_state['show_mask'] = show_mask

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
    )

gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch(share=args.share)
