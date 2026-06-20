def process_images_with_caption(raw_chunks, use_gemini=True):
    import base64
    import os
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    from unstructured.documents.elements import Image, FigureCaption

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if use_gemini and not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key) if use_gemini else None
    processed_images = []

    for idx, chunk in enumerate(raw_chunks):

        if isinstance(chunk, Image):

            caption = None
            if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
                caption = raw_chunks[idx + 1].text

            image_data = {
                "index": idx,
                "caption": caption if caption else "NO CAPTION",
                "image_text": chunk.text,
                "base64_image": chunk.metadata.image_base64,
                "content": chunk.text,
                "content_type": "image/png",
                "file_name": chunk.metadata.filename,
                "page_number": getattr(chunk.metadata, "page_number", None),
            }

            try:

                if not use_gemini:
                    fallback_parts = [image_data["image_text"], image_data["caption"]]
                    image_data["content"] = " ".join(part for part in fallback_parts if part and part != "NO CAPTION")
                    processed_images.append(image_data)
                    continue

                b64_string = image_data["base64_image"]
                if not b64_string:
                    print(f"No base64 image found for index {idx}")
                    continue

                if "base64," in b64_string:
                    b64_string = b64_string.split("base64,")[1]

                image_binary = base64.b64decode(b64_string)

                image_part = types.Part.from_bytes(
                    data=image_binary,
                    mime_type="image/png"
                )

                prompt_text = (
                    "Describe the image in detail.\n"
                    f"Context Caption: {image_data['caption']}\n"
                    f"Text in image: {image_data['image_text']}"
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[image_part, prompt_text]
                )

                image_data["content"] = response.text
                processed_images.append(image_data)

            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue

    return processed_images



def process_tables_with_caption(raw_chunks, use_gemini=True):
    from unstructured.documents.elements import Table
    import os
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if use_gemini and not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key) if use_gemini else None
    processed_table = []

    for idx, element in enumerate(raw_chunks):
        if isinstance(element, Table):

            table_data = {
                "index": idx,
                "table_as_html": element.metadata.text_as_html,
                "table_text": element.text,
                "content": element.text,
                "content_type": "table",
                "file_name": element.metadata.filename,
                "page_number": getattr(element.metadata, "page_number", None),
            }

            try:

                if not use_gemini:
                    processed_table.append(table_data)
                    continue

                prompt_text = (
                    "Analyze the table and produce a detailed caption.\n"
                    "Include structure, columns, insights, and important values.\n"
                )

                if not table_data["table_as_html"]:
                    print(f"No HTML found for table at index {idx}")
                    continue

                html_part = types.Part.from_text(
                    text=table_data["table_as_html"]
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt_text, html_part]
                )

                table_data["content"] = response.text
                processed_table.append(table_data)

            except Exception as e:
                print(f"Error generating caption for table {idx}: {str(e)}")
                continue

    return processed_table



def create_semantic_chunks(raw_chunks):
    from unstructured.documents.elements import CompositeElement

    processed_chunks = []

    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, CompositeElement):
            chunk_data = {
                "content": chunk.text,
                "content_type": "text",
                "file_name": chunk.metadata.filename,
                "page_number": getattr(chunk.metadata, "page_number", None),
            }
            processed_chunks.append(chunk_data)

    return processed_chunks



if __name__ == "__main__":
    import os
    from unstructured.partition.pdf import partition_pdf

    pdf_path = os.path.join("files", "tani.pdf")

    raw_chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None
    )

    print("===== TABLES =====")
    processed_tables = process_tables_with_caption(raw_chunks, use_gemini=True)
    for table in processed_tables:
        print(table)

    print("\n===== SEMANTIC TEXT CHUNKS =====")

    text_chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=1000,            
        combine_text_under_n_chars=200,  
        new_after_n_chars=1500 ,          
    )

    semantic_chunks = create_semantic_chunks(text_chunks)
    for chunk in semantic_chunks:
        print(chunk)
