# app/ia_infer.py
import os
import tempfile
import nibabel as nib
import numpy as np
from google.cloud import storage

# Importar configurações
from .core.config import GCP_PROJECT_ID

# --- Funções Auxiliares GCS (Reutilizadas) ---
storage_client = None

def get_storage_client():
    """Inicializa e retorna o cliente GCS."""
    global storage_client
    if storage_client is None:
        try:
            storage_client = storage.Client(project=GCP_PROJECT_ID)
            print("Cliente GCS inicializado em ia_infer.")
        except Exception as e:
            print(f"ERRO ao inicializar cliente GCS em ia_infer: {e}")
            # Retorna None ou levanta a exceção dependendo de como quer tratar
            raise ConnectionError(f"Falha ao inicializar cliente GCS: {e}")
    return storage_client

def download_blob_to_file(gcs_uri: str, destination_file_name: str):
    """Baixa um blob do GCS para um arquivo local."""
    client = get_storage_client()
    if not gcs_uri.startswith("gs://"):
        raise ValueError("URI inválida do GCS. Deve começar com gs://")
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Blob {gcs_uri} baixado para {destination_file_name}")
    except Exception as e:
        print(f"ERRO ao baixar blob {gcs_uri}: {e}")
        raise # Re-levanta a exceção para ser tratada pela função chamadora

def upload_file_to_blob(source_file_name: str, gcs_bucket_name: str, destination_blob_name: str, content_type=None):
    """Faz upload de um arquivo local para o GCS."""
    client = get_storage_client()
    try:
        bucket = client.bucket(gcs_bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name, content_type=content_type)
        gcs_path = f"gs://{gcs_bucket_name}/{destination_blob_name}"
        print(f"Arquivo {source_file_name} enviado para {gcs_path}")
        return gcs_path
    except Exception as e:
        print(f"ERRO ao fazer upload do arquivo {source_file_name} para gs://{gcs_bucket_name}/{destination_blob_name}: {e}")
        raise # Re-levanta a exceção
# --- Fim Funções Auxiliares GCS ---

def executar_inferencia_simples_gcs(
    input_nifti_gcs_path: str,
    output_gcs_bucket_name: str,
    output_gcs_folder_for_exam: str
    ) -> dict:
    """
    Executa a inferência SIMPLES (percentil 95):
    1. Baixa NIfTI de input_nifti_gcs_path.
    2. Calcula a máscara (percentil 95).
    3. Faz upload da máscara NIfTI e achados.txt para o GCS.
    Retorna um dicionário com status e caminhos GCS dos outputs.
    """
    print(f"Iniciando IA Simples para: {input_nifti_gcs_path}")
    base_input_filename = os.path.basename(input_nifti_gcs_path)
    # Remove extensões como .nii.gz ou .nii para criar nome base
    nome_base_sem_ext = base_input_filename.split('.')[0]

    output_mask_gcs_blob_name = f"{output_gcs_folder_for_exam}/{nome_base_sem_ext}_mask_simple.nii.gz"
    output_achados_gcs_blob_name = f"{output_gcs_folder_for_exam}/{nome_base_sem_ext}_achados_simple.txt"

    # Usar um diretório temporário para arquivos baixados e gerados localmente no container
    with tempfile.TemporaryDirectory() as tmpdir:
        local_input_nifti_path = os.path.join(tmpdir, base_input_filename)
        local_output_mask_path = os.path.join(tmpdir, f"{nome_base_sem_ext}_mask_simple.nii.gz")
        local_output_achados_path = os.path.join(tmpdir, f"{nome_base_sem_ext}_achados_simple.txt")

        try:
            # 1. Baixar NIfTI de entrada
            download_blob_to_file(input_nifti_gcs_path, local_input_nifti_path)

            # 2. Processamento Simples (Percentil 95)
            print(f"IA Simples: Processando {local_input_nifti_path}...")
            img_original = nib.load(local_input_nifti_path)
            # Garante que os dados sejam carregados como float para np.percentile
            dados = img_original.get_fdata(dtype=np.float32)

            # Verifica se os dados não estão vazios/constantes
            if dados.size == 0 or np.all(dados == dados.flat[0]):
                print("IA Simples: Dados de imagem vazios ou constantes.")
                mascara_numpy = np.zeros(dados.shape, dtype=np.uint8)
                limiar = np.nan # Não há limiar calculável
            else:
                # Calcula o limiar (95º percentil)
                limiar = np.percentile(dados, 95)
                # Cria a máscara binária
                mascara_numpy = (dados > limiar).astype(np.uint8)

            print(f"IA Simples: Máscara (percentil 95) calculada. Limiar={limiar:.2f if not np.isnan(limiar) else 'N/A'}")

            # 3. Salvar máscara NIfTI localmente (usando affine e header originais)
            mascara_nifti = nib.Nifti1Image(mascara_numpy, img_original.affine, img_original.header)
            nib.save(mascara_nifti, local_output_mask_path)

            # 4. Gerar arquivo de achados localmente
            achados_text = (
                f"Relatório da Análise Simples para: {base_input_filename}\n"
                f"- Tipo: Segmentação por Limiar de Intensidade (Percentil 95)\n"
                f"- Limiar de Intensidade Calculado: {limiar:.2f if not np.isnan(limiar) else 'N/A'}\n"
                f"- Máscara gerada: {os.path.basename(local_output_mask_path)}\n"
                f"- Voxels na máscara: {np.sum(mascara_numpy)}\n"
                f"- Status: Processado com sucesso (IA Simples).\n"
            )
            with open(local_output_achados_path, "w", encoding='utf-8') as f:
                f.write(achados_text)

            # 5. Fazer upload dos resultados para o GCS
            mask_gcs_path = upload_file_to_blob(local_output_mask_path, output_gcs_bucket_name, output_mask_gcs_blob_name, content_type="application/gzip")
            achados_gcs_path = upload_file_to_blob(local_output_achados_path, output_gcs_bucket_name, output_achados_gcs_blob_name, content_type="text/plain; charset=utf-8")

            print("IA Simples: Upload dos resultados concluído.")
            return {
                "success": True,
                "output_mask_gcs_path": mask_gcs_path,
                "output_achados_gcs_path": achados_gcs_path
            }

        except Exception as e:
            mensagem_erro = f"ERRO durante a IA Simples para {input_nifti_gcs_path}: {str(e)}"
            print(mensagem_erro)
            # Tenta salvar um arquivo de achados com o erro localmente
            try:
                with open(local_output_achados_path, "w", encoding='utf-8') as f_err:
                    f_err.write(mensagem_erro)
                # Faz upload do arquivo de erro para o GCS
                achados_gcs_path_erro = upload_file_to_blob(local_output_achados_path, output_gcs_bucket_name, output_achados_gcs_blob_name, content_type="text/plain; charset=utf-8")
            except Exception as e_upload_err:
                print(f"ERRO ao tentar fazer upload do log de erro da IA Simples: {e_upload_err}")
                achados_gcs_path_erro = None # Falhou ao fazer upload do log de erro

            return {
                "success": False,
                "error_message": mensagem_erro,
                "output_achados_gcs_path": achados_gcs_path_erro # Pode ser None se o upload do log falhar
            }