import hashlib
import os
from tqdm import tqdm
from dotenv import load_dotenv
from helper import EMBEDDING_DIMENSION

load_dotenv()


def create_index_if_not_exists(client, index_name, reset_index=False):
    """Create OpenSearch index with KNN configuration."""
    if client.indices.exists(index=index_name):
        if reset_index:
            print(f'Index "{index_name}" already exists. Deleting it because RESET_INDEX=true...')
            client.indices.delete(index=index_name)
        else:
            print(f'Index "{index_name}" already exists. Keeping existing documents.')
            return

    # Proper OpenSearch mappings with KNN vector
    mappings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text"
                },
                "content_type": {
                    "type": "keyword"
                },
                "file_name": {
                    "type": "keyword"
                },
                "page_number": {
                    "type": "integer"
                },
                "chunk_id": {
                    "type": "keyword"
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 16
                        }
                    }
                }
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f'✓ Index "{index_name}" created successfully.')
    except Exception as e:
        print(f"Error creating index {index_name}: {str(e)}")
        raise


def prepare_chunks_for_ingestion(chunks):
    """Prepare chunks with embeddings for ingestion into OpenSearch"""
    from helper import EMBEDDING_DIMENSION, get_embedding

    prepared_chunks = []
    skipped_count = 0

    for chunk in tqdm(chunks, desc="Preparing chunks"):
        # Extract content safely
        content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
        
        if not content or (isinstance(content, str) and not content.strip()):
            skipped_count += 1
            continue

        try:
            embedding = get_embedding(content)
            if embedding is None:
                print(f"Warning: Skipping chunk because embedding generation failed: {content[:80]}...")
                skipped_count += 1
                continue
            
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif isinstance(embedding, tuple):
                embedding = list(embedding)
            
            if len(embedding) != EMBEDDING_DIMENSION:
                print(
                    "Warning: Embedding dimension mismatch. "
                    f"Expected {EMBEDDING_DIMENSION}, got {len(embedding)}"
                )
                skipped_count += 1
                continue

            file_name = chunk.get("file_name") if isinstance(chunk, dict) else None
            content_type = chunk.get("content_type", "text") if isinstance(chunk, dict) else "text"
            page_number = chunk.get("page_number") if isinstance(chunk, dict) else None
            id_basis = f"{file_name}|{page_number}|{content_type}|{content}"
            chunk_id = hashlib.sha256(id_basis.encode("utf-8")).hexdigest()

            chunk_data = {
                "_id": chunk_id,
                "chunk_id": chunk_id,
                "content": str(content).strip(),
                "content_type": content_type,
                "file_name": file_name,
                "page_number": page_number,
                "embedding": embedding
            }
            prepared_chunks.append(chunk_data)

        except Exception as e:
            print(f"Error preparing chunk: {str(e)}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"Skipped {skipped_count} chunks due to errors or empty content.")
    
    return prepared_chunks


def ingest_chunks_to_opensearch(client, index_name, chunks, batch_size=100):
    """Ingest chunks to OpenSearch in batches"""
    from opensearchpy import helpers

    if not chunks:
        print("No chunks to ingest.")
        return

    total_chunks = len(chunks)
    print(f"Starting bulk ingestion of {total_chunks} chunks...\n")

    # Ingest in batches to show progress and avoid memory issues
    successful_docs = 0
    failed_docs = 0

    for i in tqdm(range(0, total_chunks, batch_size), desc="Ingesting batches"):
        batch = chunks[i:i + batch_size]
        
        # Create bulk actions
        actions = [
            {
                "_index": index_name,
                "_id": chunk["_id"],
                "_source": {
                    "content": chunk["content"],
                    "content_type": chunk["content_type"],
                    "file_name": chunk["file_name"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "embedding": chunk["embedding"]
                }
            }
            for chunk in batch
        ]

        try:
            # Perform bulk ingestion
            result = helpers.bulk(
                client,
                actions,
                raise_on_error=False
            )
            
            # Result is a tuple: (success_count, error_list)
            if isinstance(result, tuple):
                success_count = result[0]
                errors = result[1] if len(result) > 1 else []
                failed_count = len(errors)
            else:
                success_count = len(batch)
                failed_count = 0
            
            successful_docs += success_count
            failed_docs += failed_count

            if failed_count > 0:
                print(f"Batch {i//batch_size + 1}: {failed_count} documents failed to ingest")

        except Exception as e:
            print(f"Error ingesting batch starting at index {i}: {str(e)}")
            failed_docs += len(batch)

    # Refresh index
    try:
        client.indices.refresh(index=index_name)
    except:
        pass

    print(f"\n{'='*50}")
    print(f"Bulk ingestion completed.")
    print(f"✓ Successfully ingested: {successful_docs} documents")
    print(f"✗ Failed documents: {failed_docs} documents")
    print(f"{'='*50}\n")


def ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name):
    """Orchestrate the ingestion of all content types into OpenSearch"""
    from helper import get_opensearch_client

    # Get OpenSearch client using env/default configuration.
    client = get_opensearch_client(log_connection=True)

    # Verify client connection
    try:
        if not client.ping():
            print("Error: Cannot connect to OpenSearch")
            return
    except Exception as e:
        print(f"Error connecting to OpenSearch: {str(e)}")
        return

    reset_index = os.getenv("RESET_INDEX", "false").lower() in {"1", "true", "yes"}
    create_index_if_not_exists(client, index_name, reset_index=reset_index)

    # Prepare all chunks
    print("\nPreparing image chunks...")
    image_chunks = prepare_chunks_for_ingestion(processed_images) if processed_images else []
    
    print("\nPreparing table chunks...")
    table_chunks = prepare_chunks_for_ingestion(processed_tables) if processed_tables else []
    
    print("\nPreparing semantic chunks...")
    semantic_chunks_prepared = prepare_chunks_for_ingestion(semantic_chunks) if semantic_chunks else []

    # Combine all chunks
    all_chunks = image_chunks + table_chunks + semantic_chunks_prepared
    
    print(f"\nTotal chunks to ingest: {len(all_chunks)}")
    print(f"  - Image chunks: {len(image_chunks)}")
    print(f"  - Table chunks: {len(table_chunks)}")
    print(f"  - Semantic chunks: {len(semantic_chunks_prepared)}\n")

    if all_chunks:
        # Bulk ingestion with progress bar
        ingest_chunks_to_opensearch(client, index_name, all_chunks)
    else:
        print("No chunks to ingest.")


def ingest_pdf_file(pdf_path, index_name=None, use_gemini_captions=True):
    """Process a PDF and ingest text, table, and image chunks into OpenSearch."""
    from unstructured.partition.pdf import partition_pdf
    from chunking import process_images_with_caption, process_tables_with_caption, create_semantic_chunks

    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    index_name = index_name or os.getenv("INDEX_NAME", "pdf_content_index")

    print("Partitioning PDF for images and tables...")
    raw_chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None
    )

    print("===== Processing Images =====")
    processed_images = process_images_with_caption(raw_chunks, use_gemini=use_gemini_captions)
    print(f"Processed {len(processed_images)} images")

    print("\n===== Processing Tables =====")
    processed_tables = process_tables_with_caption(raw_chunks, use_gemini=use_gemini_captions)
    print(f"Processed {len(processed_tables)} tables")

    print("\n===== Creating Semantic Text Chunks =====")
    text_chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=1000,
        combine_text_under_n_chars=200,
        new_after_n_chars=1500
    )

    semantic_chunks = create_semantic_chunks(text_chunks)
    print(f"Created {len(semantic_chunks)} semantic chunks")

    print(f"\n===== Ingesting into OpenSearch Index '{index_name}' =====\n")
    ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name)
    return {
        "index_name": index_name,
        "pdf_path": pdf_path,
        "images": len(processed_images),
        "tables": len(processed_tables),
        "text_chunks": len(semantic_chunks),
    }


if __name__ == "__main__":
    try:
        pdf_path = os.getenv("PDF_PATH", os.path.join("files", "tani.pdf"))
        use_gemini_captions = os.getenv("USE_GEMINI_CAPTIONS", "true").lower() in {"1", "true", "yes"}
        ingest_pdf_file(pdf_path, use_gemini_captions=use_gemini_captions)
        print("✓ Ingestion pipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: PDF file not found - {str(e)}")
    except Exception as e:
        print(f"Error in ingestion pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
