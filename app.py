import os
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

import gradio as gr
from PIL import Image
import torch
import uuid
import os
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM


MAX_IMAGES = 150

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update for the captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            print(base_name)
            print(image_value)
            if base_name in txt_files_dict:
                print("entrou")
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else "[trigger]" if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    # Update prompt samples
    updates.append(gr.update(placeholder=f'A portrait of person in a bustling cafe {concept_sentence}', value=f'A person in a bustling cafe {concept_sentence}'))
    updates.append(gr.update(placeholder=f"A mountainous landscape in the style of {concept_sentence}"))
    updates.append(gr.update(placeholder=f"A {concept_sentence} in a mall"))
    updates.append(gr.update(visible=True))
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    destination_folder = str(f"fluxtrainer/datasets/{uuid.uuid4()}")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = os.path.join(destination_folder, caption_file_name)
        print(f"image_path={new_image_path}, caption_path = {caption_path}")
        with open(caption_path, 'w') as file:
            file.write(sh)

    print("destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, *captions):
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def gen_sh(
    lora_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    model_to_train,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
):
    output_name = slugify(lora_name)

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    if model_to_train == "dev":
        pretrained_model_path = "fluxtrainer/models/flux1-dev.sft"
    elif model_to_train == "schnell":
        pretrained_model_path = "fluxtrainer/models/flux1-schnell.sft"
    clip_path = "fluxtrainer/models/clip_l.safetensors"
    t5_path = "fluxtrainer/models/t5xxl_fp16.safetensors"
    ae_path = "fluxtrainer/models/ae.sft"
    output_dir = "fluxtrainer/outputs"

    ############# Optimizer args ########################

    # 24G+
    optimizer = ""

    # 16G
    optimizer = f"""
--optimizer_type adafactor
--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"
"""

    # 12G
    optimizer = f"""
--optimizer_type adafactor
--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --split_mode --network_args "train_blocks=single" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
"""
    sh = f"""
accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim 4 {line_break}
  --optimizer_type adamw8bit
  --learning_rate {learning_rate} {line_break}
  --network_train_unet_only {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config dataset.toml {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}
  {optimizer}
"""

    with open(f"fluxtrainer/train.{file_type}", 'w') as file:
        file.write(sh)

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens
):
    toml = f"""
[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = {dataset_folder}
  class_tokens = '{class_tokens}'
  num_repeats = 20
"""
    with open('fluxtrainer/dataset.toml', 'w') as file:
        file.write(toml)

def start_training(
    lora_name,
    resolution,
    seed,
    class_tokens,
#    steps,
    learning_rate,
    network_dim,
    model_to_train,
    dataset_folder,
    sample_1,
    sample_2,
    sample_3,
):
    # write custom script and toml
    os.makedirs("fluxtrainer", exist_ok=True)
    os.makedirs("fluxtrainer/models", exist_ok=True)
    os.makedirs("fluxtrainer/outputs", exist_ok=True)

    # generate accelerate script
    gen_sh(lora_name, resolution, seed, workers, learning_rate, model_to_train, max_train_epochs, save_every_n_epochs, timestep_sampling, guidance_scale)
    # generate toml
    gen_toml(dataset_folder, resolution, class_tokens)

    return f"Training completed successfully. Model saved as {slugged_lora_name}"


theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
"""
with gr.Blocks(theme=theme, css=css) as demo:
    with gr.Column() as main_ui:
        with gr.Row():
            lora_name = gr.Textbox(
                label="The name of your LoRA",
                info="This has to be a unique name",
                placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
            )
            class_tokens = gr.Textbox(
                label="Trigger word/sentence",
                info="Trigger word or sentence to be used",
                placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                interactive=True,
            )
        with gr.Group(visible=True) as image_upload:
            with gr.Row():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="Upload your images",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
                with gr.Column(scale=3, visible=False) as captioning_area:
                    with gr.Column():
                        gr.Markdown(
                            """# Custom captioning
<p style="margin-top:0">You can optionally add a custom caption for each image (or use an AI model for this). [trigger] will represent your concept sentence/trigger word.</p>
""", elem_classes="group_padding")
                        do_captioning = gr.Button("Add AI captions with Florence-2")
                        output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
        

        with gr.Accordion("Advanced options", open=False):
            #resolution = gr.Number(label="Resolution", value=512, minimum=512, maximum=1024, step=512)
            seed = gr.Number(label="Seed", value=42)
            workers = gr.Number(label="Workers", value=2)
            learning_rate = gr.Number(label="Learning Rate", value=1e-4, minimum=1e-6, maximum=1e-3, step=1e-6)
            #learning_rate = gr.Number(label="Learning Rate", value=4e-4, minimum=1e-6, maximum=1e-3, step=1e-6)

            max_train_epochs = gr.Number(label="Max Train Epochs", value=16)
            save_every_n_epochs = gr.Number(label="Save every N epochs", value=4)

            guidance_scale = gr.Number(label="Guidance Scale", value=1.0)

            timestep_sampling = gr.Textbox(label="Timestep Sampling", value="sigmoid")

#            steps = gr.Number(label="Steps", value=1000, minimum=1, maximum=10000, step=1)
            network_dim = gr.Number(label="LoRA Rank", value=4, minimum=4, maximum=128, step=4)
            model_to_train = gr.Radio(["dev", "schnell"], value="dev", label="Model to train")
            resolution = gr.Radio([512, 1024], value=512, label="Resize dataset images")

        with gr.Accordion("Sample prompts (optional)", visible=False) as sample:
            gr.Markdown(
                "Include sample prompts to test out your trained model. Don't forget to include your trigger word/sentence (optional)"
            )
            sample_1 = gr.Textbox(label="Test prompt 1")
            sample_2 = gr.Textbox(label="Test prompt 2")
            sample_3 = gr.Textbox(label="Test prompt 3")

        output_components.append(sample)
        output_components.append(sample_1)
        output_components.append(sample_2)
        output_components.append(sample_3)
        start = gr.Button("Start training", visible=False)
        output_components.append(start)
        progress_area = gr.Markdown("")

    dataset_folder = gr.State()

    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )

    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )

    images.clear(
        hide_captioning,
        outputs=[captioning_area, sample, start]
    )

    start.click(fn=create_dataset, inputs=[size, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            lora_name,
            resolution,
            seed,
            concept_sentence,
#            steps,
            learning_rate,
            network_dim,
            model_to_train,
            dataset_folder,
            sample_1,
            sample_2,
            sample_3,
        ],
        outputs=progress_area,
    )

    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
