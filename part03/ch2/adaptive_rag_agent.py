import os
import sys
import json
import re
import logging
import traceback
from datetime import datetime
from typing import TypedDict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import requests

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from langgraph.errors import GraphRecursionError
import langchainhub as hub

# ==========================================
# 1. 로깅 및 환경 설정
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Windows 환경에서 이모지 출력 시 발생하는 cp949 인코딩 에러 방지
if hasattr(sys.stdout, 'reconfigure') and sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# .env 환경변수 로드
load_dotenv()

# ==========================================
# 2. 모델 및 DB 초기화
# ==========================================
LLM_MODEL = "gemma4:e2b-it-qat"
EMBED_MODEL = "nomic-embed-text"
MAX_RETRIES = 2
CHROMA_PERSIST_DIR = "./chroma_db"

try:
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_json = llm  # 구조화 출력 실패를 대비한 동일 객체
except Exception as e:
    logger.error(f"Ollama 모델 초기화 오류: {e}")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

try:
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    # DB가 비어있을 경우 웹 문서를 로드하여 적재 (Data Ingestion)
    if len(vectorstore.get()['ids']) == 0:
        logger.info("VectorStore가 비어있습니다. 'https://wikidocs.net/268613'에서 문서를 가져와 적재합니다...")
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        try:
            # 웹 문서 로드
            loader = WebBaseLoader("https://wikidocs.net/268613")
            docs = loader.load()
            
            # 문서 분할 (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            # VectorStore에 저장
            vectorstore.add_documents(splits)
            logger.info(f"문서 적재 완료: 총 {len(splits)}개의 청크가 Chroma DB에 저장되었습니다.")
        except ImportError:
            logger.error("beautifulsoup4 패키지가 설치되지 않아 문서를 로드할 수 없습니다. 'pip install beautifulsoup4'를 실행해주세요.")
        except Exception as e:
            logger.error(f"문서 로드 및 적재 중 오류 발생: {e}")

    retriever = vectorstore.as_retriever()
except Exception as e:
    logger.error(f"Chroma DB 초기화 실패: {e}")

# ==========================================
# 3. Pydantic 스키마 및 GraphState
# ==========================================
class RouteQuery(BaseModel):
    """질문 라우팅 목적의 스키마"""
    datasource: Literal["vectorstore", "web_search", "llm"] = Field(...)

class GradeDocument(BaseModel):
    """문서 관련성 평가 목적의 스키마"""
    binary_score: Literal["yes", "no"] = Field(...)

class GradeHallucination(BaseModel):
    """환각 평가 목적의 스키마"""
    binary_score: Literal["yes", "no"] = Field(...)

class GradeAnswer(BaseModel):
    """답변 적절성 평가 목적의 스키마"""
    binary_score: Literal["yes", "no"] = Field(...)

class GraphState(TypedDict, total=False):
    """Adaptive RAG Agent 상태 관리"""
    question: str
    original_question: str
    generation: str
    documents: List[Document]
    web_results: str
    route: Optional[Literal["vectorstore", "web_search", "llm"]]
    retry_count: int
    max_retries: int
    error: Optional[str]

# ==========================================
# 4. 구조화 출력 파싱 Fallback
# ==========================================
T = TypeVar('T', bound=BaseModel)

def parse_with_fallback(text: str, pydantic_class: Type[T]) -> T:
    """JSON 파싱 실패 시 텍스트 내 중괄호 블록을 추출하여 파싱 시도"""
    try:
        parsed_json = json.loads(text)
        return pydantic_class(**parsed_json)
    except (json.JSONDecodeError, ValidationError):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
                return pydantic_class(**parsed_json)
            except (json.JSONDecodeError, ValidationError):
                pass
                
    logger.warning(f"구조화된 파싱 실패, 기본값 반환: {text}")
    if pydantic_class == RouteQuery:
        return pydantic_class(datasource="llm")
    else:
        return pydantic_class(binary_score="yes")

# ==========================================
# 5. 프롬프트 및 체인 정의
# ==========================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 사용자의 질문을 라우팅하는 전문가입니다.
- LangChain, RAG, Agent, 기술 질문 -> 'vectorstore'
- 최신 뉴스, 오늘 날씨, 최근 동향 -> 'web_search'
- 단순 인사말, 일반 대화 -> 'llm'
출력: {{"datasource": "vectorstore" | "web_search" | "llm"}}"""),
    ("human", "{question}")
])
router_chain = router_prompt | llm_json | StrOutputParser()

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "검색된 문서가 질문과 관련있으면 'yes', 없으면 'no'. 출력: {{\"binary_score\": \"yes\" | \"no\"}}"),
    ("human", "문서: \n\n {document} \n\n 질문: {question}")
])
grader_chain = grader_prompt | llm_json | StrOutputParser()

try:
    rag_prompt = hub.pull("rlm/rag-prompt")
except Exception as e:
    logger.info("LangSmith Hub 접근 실패, 로컬 프롬프트 사용")
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "문맥(Context)을 사용하여 질문에 답변하세요. 모르면 모른다고 하세요. 3문장 이내 한국어 답변."),
        ("human", "문맥: {context}\n\n질문: {question}")
    ])
rag_chain = rag_prompt | llm | StrOutputParser()

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "답변이 문서 내용에 완전히 기반하면 'yes', 지어낸 사실이 있으면 'no'. 출력: {{\"binary_score\": \"yes\" | \"no\"}}"),
    ("human", "문서: \n\n {documents} \n\n 답변: {generation}")
])
hallucination_chain = hallucination_prompt | llm_json | StrOutputParser()

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "답변이 원 질문에 적절하면 'yes', 아니면 'no'. 출력: {{\"binary_score\": \"yes\" | \"no\"}}"),
    ("human", "질문: \n\n {question} \n\n 답변: {generation}")
])
answer_grader_chain = answer_grader_prompt | llm_json | StrOutputParser()

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", "검색에 적합하도록 키워드 중심으로 질문을 명확히 재작성하세요. 한국어로 질문 텍스트만 출력하세요."),
    ("human", "원래 질문: {question}")
])
rewriter_chain = rewriter_prompt | llm | StrOutputParser()

llm_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "친절한 AI 어시스턴트입니다. 친절하고 명확하게 한국어로 답변하세요."),
    ("human", "{question}")
])
llm_answer_chain = llm_answer_prompt | llm | StrOutputParser()

# ==========================================
# 6. Nodes (노드 함수)
# ==========================================
def retrieve_node(state: GraphState) -> GraphState:
    """문서를 검색하여 documents에 저장"""
    logger.info("---NODE: retrieve_node---")
    question = state.get("question", "")
    
    try:
        docs = retriever.invoke(question)
        if not docs:
            return {"documents": [], "error": "관련 문서를 찾지 못했습니다."}
        return {"documents": docs, "error": None}
    except Exception as e:
        logger.error(f"문서 검색 중 오류: {e}")
        return {"documents": [], "error": str(e)}

def grade_documents_node(state: GraphState) -> GraphState:
    """문서 관련성 평가"""
    logger.info("---NODE: grade_documents_node---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    filtered_docs = []
    for doc in documents:
        try:
            result_str = grader_chain.invoke({"question": question, "document": doc.page_content})
            result = parse_with_fallback(result_str, GradeDocument)
            if result.binary_score == "yes":
                filtered_docs.append(doc)
        except Exception as e:
            logger.error(f"문서 평가 실패: {e}")
            
    return {"documents": filtered_docs}

def transform_query_node(state: GraphState) -> GraphState:
    """질문 재작성"""
    logger.info("---NODE: transform_query_node---")
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)
    
    try:
        new_question = rewriter_chain.invoke({"question": question})
        if not new_question or new_question.strip() == "":
            new_question = question
        return {"question": new_question, "retry_count": retry_count + 1}
    except Exception as e:
        logger.error(f"질문 재작성 오류: {e}")
        return {"question": question, "retry_count": retry_count + 1}

def generate_node(state: GraphState) -> GraphState:
    """검색 문서를 기반으로 답변 생성"""
    logger.info("---NODE: generate_node---")
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    if not documents:
        return {"generation": "참고할 문서가 없어 답변할 수 없습니다.", "error": "문서 없음"}
        
    try:
        context = "\n".join([d.page_content for d in documents])
        generation = rag_chain.invoke({"context": context, "question": question})
        return {"generation": generation}
    except Exception as e:
        logger.error(f"답변 생성 오류: {e}")
        return {"generation": "답변 생성에 실패했습니다.", "error": str(e)}

def generate_llm_node(state: GraphState) -> GraphState:
    """외부 검색 없이 LLM 직접 답변"""
    logger.info("---NODE: generate_llm_node---")
    question = state.get("question", "")
    
    try:
        generation = llm_answer_chain.invoke({"question": question})
        return {"generation": generation}
    except Exception as e:
        logger.error(f"LLM 답변 생성 오류: {e}")
        return {"generation": "답변 생성에 실패했습니다.", "error": str(e)}

def web_search_node(state: GraphState) -> GraphState:
    """Naver API 웹 검색"""
    logger.info("---NODE: web_search_node---")
    question = state.get("question", "")
    
    today = datetime.now().strftime("%Y년 %m월 %d일")
    search_query = f"{question} ({today} 기준)"
    
    client_id = os.environ.get("NAVER_CLIENT_ID", "")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET", "")
    
    if not client_id or not client_secret:
        return {"web_results": "", "error": "네이버 검색 API 키 누락"}
    
    url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=3"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results_str = ""
        for item in data.get("items", []):
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            desc = item.get("description", "").replace("<b>", "").replace("</b>", "")
            results_str += f"제목: {title}\n요약: {desc}\n\n"
            
        if not results_str:
            return {"web_results": "", "error": "검색 결과 없음"}
            
        return {"web_results": results_str, "error": None}
    except Exception as e:
        logger.error(f"웹 검색 오류: {e}")
        return {"web_results": "", "error": "웹 검색 오류"}

def generate_web_search_node(state: GraphState) -> GraphState:
    """웹 검색 결과를 기반으로 답변 생성"""
    logger.info("---NODE: generate_web_search_node---")
    question = state.get("question", "")
    web_results = state.get("web_results", "")
    
    if not web_results:
        error_msg = state.get("error", "웹 검색 결과를 불러오지 못했습니다.")
        return {"generation": f"답변할 수 없습니다. 사유: {error_msg}"}
        
    try:
        generation = rag_chain.invoke({"context": web_results, "question": question})
        return {"generation": generation}
    except Exception as e:
        logger.error(f"웹 기반 생성 오류: {e}")
        return {"generation": "웹 결과를 요약하는데 실패했습니다.", "error": str(e)}

# ==========================================
# 7. Edges (조건부 라우팅)
# ==========================================
def route_question_edge(state: GraphState) -> str:
    logger.info("---EDGE: route_question_edge---")
    question = state.get("question", "")
    try:
        result_str = router_chain.invoke({"question": question})
        result = parse_with_fallback(result_str, RouteQuery)
        route = result.datasource
    except Exception:
        route = "llm"
    
    logger.info(f"라우팅 결과: {route}")

    if route == "vectorstore":
        return "vectorstore"
    elif route == "web_search":
        return "web_search"
    else:
        return "llm"

def decide_to_generate_edge(state: GraphState) -> str:
    logger.info("---EDGE: decide_to_generate_edge---")
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    
    if documents:
        return "generate"
    elif retry_count < MAX_RETRIES:
        return "transform_query"
    else:
        return "end"

def grade_generation_edge(state: GraphState) -> str:
    logger.info("---EDGE: grade_generation_edge---")
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)
    
    if "참고할 문서가 없어" in generation or "답변 생성에 실패" in generation:
        return "end"

    docs_text = "\n".join([d.page_content for d in documents])
    
    try:
        hallucination_str = hallucination_chain.invoke({"documents": docs_text, "generation": generation})
        hallucination_res = parse_with_fallback(hallucination_str, GradeHallucination)
        
        if hallucination_res.binary_score == "no":
            if retry_count < MAX_RETRIES:
                return "not supported"
            else:
                return "end"
            
        answer_str = answer_grader_chain.invoke({"question": state.get("original_question", ""), "generation": generation})
        answer_res = parse_with_fallback(answer_str, GradeAnswer)
        
        if answer_res.binary_score == "yes":
            return "useful"
        else:
            if retry_count < MAX_RETRIES:
                return "not useful"
            else:
                return "end"
    except Exception as e:
        logger.error(f"평가 엣지 오류: {e}")
        return "end"

# ==========================================
# 8. Graph Build
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("transform_query_node", transform_query_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_web_search_node", generate_web_search_node)
workflow.add_node("generate_llm_node", generate_llm_node)

workflow.set_conditional_entry_point(
    route_question_edge,
    {
        "vectorstore": "retrieve_node",
        "web_search": "web_search_node",
        "llm": "generate_llm_node"
    }
)

workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_conditional_edges(
    "grade_documents_node",
    decide_to_generate_edge,
    {
        "generate": "generate_node",
        "transform_query": "transform_query_node",
        "end": END
    }
)
workflow.add_conditional_edges(
    "generate_node",
    grade_generation_edge,
    {
        "useful": END,
        "not useful": "transform_query_node",
        "not supported": "generate_node",
        "end": END
    }
)
workflow.add_edge("transform_query_node", "retrieve_node")
workflow.add_edge("web_search_node", "generate_web_search_node")
workflow.add_edge("generate_web_search_node", END)
workflow.add_edge("generate_llm_node", END)

app = workflow.compile()

try:
    with open("graph_mermaid.txt", "w", encoding="utf-8") as f:
        f.write(app.get_graph().draw_mermaid())
except Exception:
    pass

# ==========================================
# 9. Main CLI
# ==========================================
def main():
    print("=" * 50)
    print("Adaptive RAG 에이전트를 시작합니다.")
    print("종료하려면 quit 또는 q를 입력하세요.")
    print("=" * 50)
    
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    while True:
        try:
            user_input = input("\n사용자: ").strip()
            if user_input.lower() in ["quit", "q", "exit"]:
                print("에이전트를 종료합니다. 이용해 주셔서 감사합니다!")
                break
            if not user_input:
                continue
                
            initial_state = {
                "question": user_input,
                "original_question": user_input,
                "generation": "",
                "documents": [],
                "web_results": "",
                "route": None,
                "retry_count": 0,
                "max_retries": MAX_RETRIES,
                "error": None
            }
            
            config = {"recursion_limit": 15}
            
            try:
                final_state = app.invoke(initial_state, config=config)
                
                final_generation = final_state.get("generation")
                if not final_generation or final_generation.strip() == "":
                    final_generation = final_state.get("error", "생성된 답변이 없습니다.")
                    
                print(f"\n[에이전트 답변]\n{final_generation}")
                
            except GraphRecursionError:
                print("\n[오류] 재시도 횟수를 초과했거나 무한 루프가 발생했습니다 (GraphRecursionError).")
            except requests.exceptions.ConnectionError as e:
                if "11434" in str(e):
                    print("\n[오류] Ollama 서비스에 연결할 수 없습니다. 프로그램이 실행 중인지 확인하세요.")
                else:
                    print(f"\n[오류] 네트워크 연결 오류 (API 통신 실패 등): {e}")
            except requests.exceptions.HTTPError as e:
                err_str = str(e)
                if e.response is not None and e.response.status_code == 404 and "11434" in err_str:
                    print(f"\n[오류] Ollama 모델을 찾을 수 없습니다. 모델 이름({LLM_MODEL})을 확인하세요.")
                elif e.response is not None and e.response.status_code == 401 and "naver" in err_str:
                    print("\n[오류] Naver API 인증 실패. .env 파일의 클라이언트 ID와 시크릿을 확인하세요.")
                else:
                    print(f"\n[오류] HTTP 통신 오류: {e}")
            except ValidationError:
                print("\n[오류] 구조화 출력(JSON) 파싱 중 오류가 발생했습니다.")
            except Exception as e:
                err_str = str(e).lower()
                if "chroma" in err_str or "embed" in err_str:
                    print(f"\n[오류] VectorStore(Chroma) 또는 임베딩 모델 오류: {e}")
                else:
                    print(f"\n[오류] 실행 중 알 수 없는 예외가 발생했습니다: {e}")
                    
                if log_level == "DEBUG":
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n입력이 취소되었습니다. 에이전트를 종료합니다.")
            break

if __name__ == "__main__":
    main()
