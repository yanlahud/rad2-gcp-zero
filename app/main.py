# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Optional
import uuid
import os
import tempfile
from datetime import datetime, timezone # Usar UTC
from fpdf import FPDF # Usando fpdf2 (a biblioteca)

from google.cloud import storage, firestore
import google.auth.exceptions # Para tratar erros de autenticação

# Importar configurações e lógica de IA
from .core.config import (
    GCP_PROJECT_ID,
    GCS_UPLOAD_BUCKET,
    GCS_OUTPUT_BUCKET,
    FIRESTORE_COLLECTION_EXAMS
)
# Importa a função de inferência SIMPLES
from .ia_infer import executar_inferencia_simples_gcs

# --- Inicialização de Clientes GCP ---
# Estes clientes serão reutilizados pelas funções da API
storage_client = None
db = None
GCP_AUTHENTICATED = False
try:
    print("Tentando inicializar clientes GCP...")
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    db = firestore.Client(project=GCP_PROJECT_ID)
    # Testa a conexão listando buckets (requer permissão storage.buckets.list que a conta tem)
    print("Testando conexão GCS...")
    storage_client.list_buckets(max_results=1)
    # Testa a conexão Firestore
    print("Testando conexão Firestore...")
    db.collection(FIRESTORE_COLLECTION_EXAMS).limit(1).get()
    GCP_AUTHENTICATED = True
    print("Clientes GCS e Firestore inicializados e autenticados com sucesso.")
except google.auth.exceptions.DefaultCredentialsError as e:
     print(f"ERRO FATAL DE AUTENTICAÇÃO GCP: Verifique se a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS está configurada corretamente OU se a conta de serviço do Cloud Run tem as permissões necessárias. Detalhe: {e}")
except Exception as e:
    print(f"ERRO FATAL AO INICIALIZAR CLIENTES GCP: {e}. A aplicação pode não funcionar corretamente.")
# --- Fim Inicialização ---

app = FastAPI(
    title="RAD2 Tomografia API (GCP - IA Simples)",
    description="API para upload e análise simples de imagens CBCT usando GCS e Firestore.",
    version="0.1.0"
)

# --- Funções Auxiliares (movidas ou usando as de ia_infer) ---
# Reutilizaremos as funções get_storage_client, download_blob_to_file, upload_file_to_blob
# de ia_infer para evitar duplicação, mas precisamos importá-las aqui também
# ou redefini-las se preferir manter os módulos separados.
# Para simplicidade, vamos apenas chamar as funções de ia_infer quando necessário no laudo.
from .ia_infer import download_blob_to_file as download_blob_from_ia_infer
from .ia_infer import upload_file_to_blob as upload_file_from_ia_infer
# --- Fim Funções Auxiliares ---

# --- Checagem de Dependência ---
# Função simples para checar se os clientes GCP estão ok
def get_db_client():
    if not db or not GCP_AUTHENTICATED:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Conexão com Firestore falhou na inicialização.")
    return db

def get_gcs_client():
    if not storage_client or not GCP_AUTHENTICATED:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Conexão com GCS falhou na inicialização.")
    return storage_client
# --- Fim Checagem de Dependência ---

@app.get("/health", summary="Verifica a saúde da aplicação e conectividade", tags=["Status"])
async def health_check():
    """Verifica a conectividade com GCS e Firestore."""
    gcs_status = "conectado" if GCP_AUTHENTICATED else "NÃO CONECTADO/AUTENTICADO"
    firestore_status = "conectado" if GCP_AUTHENTICATED else "NÃO CONECTADO/AUTENTICADO"
    ia_status = "simples_ativa"
    return {
        "status": "ok",
        "message": "API está funcionando!",
        "ia_status": ia_status,
        "gcs_status": gcs_status,
        "firestore_status": firestore_status
    }

@app.post("/upload-cbct/", summary="Faz upload de arquivos CBCT e cria registro no Firestore", status_code=201, tags=["Exames"])
async def upload_cbct_endpoint(
    files: List[UploadFile] = File(..., description="Lista de arquivos NIfTI ou DICOM a serem enviados."),
    gcs: storage.Client = Depends(get_gcs_client),
    fs: firestore.Client = Depends(get_db_client)
):
    """
    Recebe um ou mais arquivos, salva no Google Cloud Storage sob um novo ID de exame,
    e cria um registro correspondente no Firestore com status 'uploaded'.
    Identifica heuristicamente o primeiro arquivo NIfTI como principal.
    """
    exam_id = str(uuid.uuid4())
    gcs_file_paths = []
    original_filenames = []
    principal_nifti_gcs_path = None
    upload_errors = []

    print(f"Iniciando upload para novo exame: {exam_id}")

    for file_upload in files:
        print(f"Processando upload: {file_upload.filename}")
        try:
            blob_name = f"{exam_id}/{file_upload.filename}"
            bucket = gcs.bucket(GCS_UPLOAD_BUCKET)
            blob = bucket.blob(blob_name)

            contents = await file_upload.read() # Lê para memória
            blob.upload_from_string(contents, content_type=file_upload.content_type) # Upload direto
            current_gcs_path = f"gs://{GCS_UPLOAD_BUCKET}/{blob_name}"
            gcs_file_paths.append(current_gcs_path)
            original_filenames.append(file_upload.filename)
            print(f"Upload de {file_upload.filename} para {current_gcs_path} concluído.")

            # Identifica o NIfTI principal
            if file_upload.filename.lower().endswith((".nii", ".nii.gz")) and principal_nifti_gcs_path is None:
                principal_nifti_gcs_path = current_gcs_path
                print(f"Arquivo NIfTI principal identificado: {current_gcs_path}")

        except Exception as e:
            print(f"ERRO durante upload GCS do arquivo {file_upload.filename}: {str(e)}")
            # Poderia coletar erros por arquivo, mas por ora vamos falhar o request inteiro se um falhar
            raise HTTPException(status_code=500, detail=f"Erro no upload GCS do arquivo {file_upload.filename}: {str(e)}")
        # Não precisamos fechar file_upload explicitamente aqui

    if not gcs_file_paths:
         raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado.")

    # Criar registro no Firestore
    exam_doc_ref = fs.collection(FIRESTORE_COLLECTION_EXAMS).document(exam_id)
    exam_data = {
        "exam_id": exam_id,
        "created_at": datetime.now(timezone.utc),
        "status": "uploaded",
        "original_filenames": original_filenames,
        "gcs_raw_file_paths": gcs_file_paths,
        "gcs_principal_nifti_path": principal_nifti_gcs_path,
        # Inicializa campos que serão preenchidos depois
        "updated_at": None,
        "gcs_output_mask_path": None,
        "gcs_output_findings_path": None,
        "gcs_laudo_pdf_path": None,
        "error_message": None,
    }
    try:
        exam_doc_ref.set(exam_data)
        print(f"Registro Firestore criado para exame {exam_id}.")
    except Exception as e:
        print(f"ERRO ao criar registro Firestore para {exam_id}: {str(e)}")
        # Idealmente, tentaríamos limpar os arquivos do GCS aqui, mas é complexo.
        raise HTTPException(status_code=500, detail=f"Erro ao criar registro do exame no Firestore: {str(e)}")

    # Retorna 201 Created por padrão se tudo der certo
    return {"exam_id": exam_id, "message": "Upload concluído e registro criado.", "data": exam_data}

async def run_simple_analysis_background(exam_id: str, principal_nifti_gcs_path: str):
    """Função para executar a análise SIMPLES em background."""
    # Re-obter clientes dentro da tarefa background pode ser mais seguro em alguns cenários
    try:
         local_fs = firestore.Client(project=GCP_PROJECT_ID)
         exam_doc_ref_bg = local_fs.collection(FIRESTORE_COLLECTION_EXAMS).document(exam_id)
    except Exception as e:
         print(f"BG TASK ERRO FATAL ({exam_id}): Falha ao conectar ao Firestore: {e}")
         return # Não pode fazer nada sem Firestore

    print(f"BG TASK ({exam_id}): Iniciando análise SIMPLES para {principal_nifti_gcs_path}")

    # Atualiza status para processando
    try:
        exam_doc_ref_bg.update({"status": "processing_ia_simple", "updated_at": firestore.SERVER_TIMESTAMP}) # Usa timestamp do servidor
    except Exception as e_fs:
        print(f"BG TASK ERRO ({exam_id}): Falha ao atualizar status inicial no Firestore: {e_fs}")
        return # Aborta se não conseguir atualizar

    # Chama a lógica de inferência (que usa seu próprio cliente GCS)
    try:
        results = executar_inferencia_simples_gcs(
            input_nifti_gcs_path=principal_nifti_gcs_path,
            output_gcs_bucket_name=GCS_OUTPUT_BUCKET,
            output_gcs_folder_for_exam=exam_id
        )
        # Prepara dados para atualizar Firestore
        update_data = {"updated_at": firestore.SERVER_TIMESTAMP}
        if results.get("success"):
            update_data["status"] = "processed_success_simple"
            update_data["gcs_output_mask_path"] = results.get("output_mask_gcs_path")
            update_data["gcs_output_findings_path"] = results.get("output_achados_gcs_path")
            update_data["error_message"] = None # Limpa erro anterior se houver
        else:
            update_data["status"] = "processed_error_simple"
            update_data["error_message"] = results.get("error_message", "Erro desconhecido na IA simples.")
            # Mesmo com erro, pode haver um path para o log de achados
            if results.get("output_achados_gcs_path"):
                update_data["gcs_output_findings_path"] = results.get("output_achados_gcs_path")

    except Exception as e_infer:
        # Erro na própria função executar_inferencia_simples_gcs
        print(f"BG TASK ERRO FATAL ({exam_id}) durante execução da inferência: {str(e_infer)}")
        update_data = {
            "status": "error_processing_fatal",
            "error_message": f"Erro fatal na análise: {str(e_infer)}",
            "updated_at": firestore.SERVER_TIMESTAMP
            }

    # Atualiza Firestore com o resultado final
    try:
        exam_doc_ref_bg.update(update_data)
        print(f"BG TASK ({exam_id}): Análise SIMPLES concluída. Status final: {update_data['status']}")
    except Exception as e_fs_final:
        print(f"BG TASK ERRO FATAL ({exam_id}): Falha ao atualizar status final no Firestore: {e_fs_final}")

@app.post("/ia-analisar/{exam_id}", summary="Inicia a análise de IA SIMPLES (em background)", status_code=202, tags=["Análise"])
async def analisar_cbct_endpoint(
    exam_id: str,
    background_tasks: BackgroundTasks,
    fs: firestore.Client = Depends(get_db_client) # Injeta dependência Firestore
    ):
    """
    Verifica o status do exame no Firestore e, se apropriado ('uploaded'),
    agenda a tarefa de análise simples para rodar em background.
    Retorna imediatamente com status 202 Accepted.
    """
    print(f"Recebida requisição de análise para exame: {exam_id}")
    exam_doc_ref = fs.collection(FIRESTORE_COLLECTION_EXAMS).document(exam_id)
    try:
        exam_snapshot = exam_doc_ref.get()
        if not exam_snapshot.exists:
            print(f"Exame {exam_id} não encontrado no Firestore.")
            raise HTTPException(status_code=404, detail=f"Exame com ID {exam_id} não encontrado.")
        exam_data = exam_snapshot.to_dict()
    except Exception as e:
         print(f"Erro ao buscar exame {exam_id} no Firestore: {str(e)}")
         raise HTTPException(status_code=500, detail=f"Erro ao buscar dados do exame no Firestore: {str(e)}")

    current_status = exam_data.get("status")
    print(f"Exame {exam_id} - Status atual: {current_status}")

    # Definição clara de status
    allowed_initial_status = "uploaded"
    processing_status = "processing_ia_simple"
    final_statuses = ["processed_success_simple", "processed_error_simple", "error_no_nifti_path", "error_processing_fatal"]

    if current_status == processing_status:
         print(f"Análise para {exam_id} já está em progresso.")
         return {"exam_id": exam_id, "message": "Análise já está em progresso."} # Retorna 200 OK (ou mantém 202)
    if current_status in final_statuses:
         print(f"Análise para {exam_id} não iniciada, estado final: {current_status}.")
         raise HTTPException(status_code=409, detail=f"Exame já processado ou em estado final: {current_status}.")
    if current_status != allowed_initial_status:
         print(f"Análise para {exam_id} não iniciada, estado inválido: {current_status}.")
         raise HTTPException(status_code=400, detail=f"Exame em estado inválido para iniciar análise: {current_status}")

    principal_nifti_gcs_path = exam_data.get("gcs_principal_nifti_path")
    if not principal_nifti_gcs_path or not principal_nifti_gcs_path.startswith("gs://"):
        error_msg = "Caminho GCS do arquivo NIfTI principal inválido ou não encontrado nos metadados."
        print(f"Erro para {exam_id}: {error_msg}")
        try: # Tenta atualizar o status no Firestore antes de levantar o erro
             exam_doc_ref.update({"status": "error_no_nifti_path", "updated_at": firestore.SERVER_TIMESTAMP})
        except Exception as e_fs_update:
             print(f"Erro ao atualizar status de erro no Firestore para {exam_id}: {e_fs_update}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Adiciona a tarefa em background
    print(f"Agendando análise em background para {exam_id} com arquivo {principal_nifti_gcs_path}")
    background_tasks.add_task(run_simple_analysis_background, exam_id, principal_nifti_gcs_path)

    # Retorna imediatamente 202 Accepted
    return {"exam_id": exam_id, "message": "Análise SIMPLES iniciada em background. Verifique o status."}

@app.get("/status/{exam_id}", summary="Verifica o status e metadados de um exame", tags=["Status"])
async def status_exame_endpoint(
    exam_id: str,
    fs: firestore.Client = Depends(get_db_client)
    ):
    """Retorna os dados do documento do exame armazenado no Firestore."""
    print(f"Buscando status para exame: {exam_id}")
    exam_doc_ref = fs.collection(FIRESTORE_COLLECTION_EXAMS).document(exam_id)
    try:
        exam_snapshot = exam_doc_ref.get()
        if not exam_snapshot.exists:
            print(f"Exame {exam_id} não encontrado.")
            raise HTTPException(status_code=404, detail=f"Exame com ID {exam_id} não encontrado.")
        print(f"Status encontrado para {exam_id}.")
        return exam_snapshot.to_dict()
    except Exception as e:
        print(f"Erro ao buscar status para {exam_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao ler dados do Firestore: {str(e)}")

@app.get("/laudo/{exam_id}", summary="Gera e retorna o laudo em PDF (IA Simples)", tags=["Laudos"])
async def gerar_laudo_endpoint(
    exam_id: str,
    fs: firestore.Client = Depends(get_db_client) # Injeta Firestore
    # GCS client será obtido pela função de download
    ):
    """
    Busca os dados do exame no Firestore, verifica o status, baixa o arquivo de achados
    do GCS, gera um PDF e o retorna.
    """
    print(f"Requisição de laudo para exame: {exam_id}")
    exam_doc_ref = fs.collection(FIRESTORE_COLLECTION_EXAMS).document(exam_id)
    try:
        exam_snapshot = exam_doc_ref.get()
        if not exam_snapshot.exists:
             print(f"Laudo: Exame {exam_id} não encontrado.")
             raise HTTPException(status_code=404, detail=f"Exame com ID {exam_id} não encontrado.")
        exam_data = exam_snapshot.to_dict()
    except Exception as e:
         print(f"Laudo: Erro ao buscar exame {exam_id} no Firestore: {str(e)}")
         raise HTTPException(status_code=500, detail=f"Erro ao buscar dados do exame no Firestore: {str(e)}")

    # Verifica status
    if exam_data.get("status") != "processed_success_simple":
         print(f"Laudo: Exame {exam_id} não está no estado correto (status: {exam_data.get('status')})")
         raise HTTPException(status_code=400, detail=f"Laudo não pode ser gerado. Status do exame: {exam_data.get('status')}")

    achados_gcs_path = exam_data.get("gcs_output_findings_path")
    if not achados_gcs_path or not achados_gcs_path.startswith("gs://"):
         print(f"Laudo: Caminho do arquivo de achados inválido para {exam_id}: {achados_gcs_path}")
         raise HTTPException(status_code=404, detail="Caminho do arquivo de achados inválido ou não encontrado nos metadados.")

    pdf_filename = f"laudo_simples_{exam_id}.pdf"

    # Usar diretório temporário seguro
    with tempfile.TemporaryDirectory() as tmpdir:
        local_achados_path = os.path.join(tmpdir, "achados.txt")
        local_laudo_pdf_path = os.path.join(tmpdir, pdf_filename)

        try:
            # 1. Baixar arquivo de achados do GCS
            print(f"Laudo: Baixando {achados_gcs_path}...")
            download_blob_from_ia_infer(achados_gcs_path, local_achados_path)

            # 2. Gerar PDF
            print(f"Laudo: Gerando PDF para {exam_id}...")
            pdf = FPDF()
            pdf.add_page()
            # Adicionar fonte que suporte UTF-8 (necessário no container Docker)
            # Tenta encontrar uma fonte comum. Se falhar, usa a padrão.
            try:
                # Caminhos comuns em containers Linux baseados em Debian/Ubuntu
                font_path_dejavu = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
                if os.path.exists(font_path_dejavu):
                     pdf.add_font('DejaVu', '', font_path_dejavu, uni=True)
                     pdf.set_font("DejaVu", "", 12)
                     print("Usando fonte DejaVu para PDF.")
                else:
                     pdf.set_font("Arial", "", 12) # Fallback
                     print("Fonte DejaVu não encontrada, usando Arial (pode ter problemas com UTF-8).")
            except RuntimeError:
                 pdf.set_font("Arial", "", 12) # Fallback em caso de erro na fonte
                 print("Erro ao adicionar fonte DejaVu, usando Arial.")


            pdf.set_font("Arial", "B", 16) # Usa Arial para o título
            pdf.cell(0, 10, f"Laudo Tomografia CBCT (IA Simples)", 0, 1, "C")
            pdf.set_font("Arial", "", 10) # Fonte menor para ID
            pdf.cell(0, 10, f"Exame ID: {exam_id}", 0, 1, "C")
            pdf.ln(10)

            pdf.set_font(pdf.font_family, "", 12) # Volta para a fonte principal (DejaVu ou Arial)
            try:
                 with open(local_achados_path, "r", encoding='utf-8') as f_achados:
                    for linha in f_achados:
                       pdf.multi_cell(0, 8, txt=linha.strip()) # Ajusta altura da linha
            except Exception as e_pdf_text:
                 print(f"Laudo: Erro ao adicionar texto dos achados ao PDF: {e_pdf_text}")
                 pdf.multi_cell(0, 10, txt="Erro ao ler detalhes dos achados.")

            pdf.output(local_laudo_pdf_path, "F")
            print(f"Laudo: PDF gerado em {local_laudo_pdf_path}")

            # 3. Retorna o arquivo PDF gerado
            return FileResponse(path=local_laudo_pdf_path, media_type='application/pdf', filename=pdf_filename)

        except FileNotFoundError:
             print(f"Laudo: Erro - Arquivo de achados não encontrado localmente após tentativa de download de {achados_gcs_path}")
             raise HTTPException(status_code=404, detail="Arquivo de achados não encontrado no GCS.")
        except Exception as e:
            print(f"Laudo: Erro detalhado ao gerar laudo para {exam_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Erro interno ao gerar o laudo PDF: {str(e)}")