import os
import sys
import json
import re
import logging
from datetime import datetime
from typing import TypedDict, List, Literal, Optional, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
import langchainhub as hub
import requests

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. 설정 및 초기화
# ==========================================
LLM_MODEL = "gemma4:e2b-it-qat"
EMBED_MODEL = "nomic-embed-text"

try:
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_json = llm
except Exception as e:
    logger.error(f"Ollama 모델 초기화 중 오류 발생: {e}")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

CHROMA_PERSIST_DIR = "./chroma_db"
try:
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever()
except Exception as e:
    logger.error(f"Chroma DB 초기화 실패: {e}")

# ==========================================
# 2. Pydantic 스키마 및 State 정의
# ==========================================
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search", "llm"] = Field(...)

class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)

class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)

class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)

class GraphState(TypedDict, total=False):
    """
    Adaptive RAG Agent 상태 관리를 위한 TypedDict
    기본값이 필요한 경우를 대비해 total=False를 사용합니다.
    """
    question: str
    original_question: str
    generation: str
    documents: List[Document]
    web_results: str  # 네이버 검색 결과를 문자열로 저장
    route: Optional[Literal["vectorstore", "web_search", "llm"]]
    retry_count: int
    max_retries: int
    error: Optional[str]

# ==========================================
# 3. 구조화 출력 Fallback
# ==========================================
T = TypeVar('T', bound=BaseModel)

def parse_with_fallback(text: str, pydantic_class: Type[T]) -> T:
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
# 4. Chains
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
# 5. Nodes
# ==========================================

def retrieve_node(state: GraphState) -> GraphState:
    """
    Retriever를 사용하여 문서를 검색하고 State의 documents 필드에 저장합니다.
    """
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
    """
    검색된 문서들의 관련성을 평가하여 'yes'인 문서만 필터링합니다.
    """
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
            logger.error(f"문서 평가 실패 (제외 처리됨): {e}")
            
    return {"documents": filtered_docs}

def transform_query_node(state: GraphState) -> GraphState:
    """
    검색 성능 향상을 위해 사용자의 질문을 재작성합니다.
    """
    logger.info("---NODE: transform_query_node---")
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)
    
    try:
        new_question = rewriter_chain.invoke({"question": question})
        # 재작성 결과가 비어있으면 기존 질문 유지
        if not new_question or new_question.strip() == "":
            new_question = question
            
        return {"question": new_question, "retry_count": retry_count + 1}
    except Exception as e:
        logger.error(f"질문 재작성 중 오류: {e}")
        return {"question": question, "retry_count": retry_count + 1}

def generate_node(state: GraphState) -> GraphState:
    """
    검색된 VectorStore 문서를 문맥으로 RAG 답변을 생성합니다.
    """
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
        logger.error(f"답변 생성 중 오류: {e}")
        return {"generation": "답변 생성에 실패했습니다.", "error": str(e)}

def generate_llm_node(state: GraphState) -> GraphState:
    """
    외부 검색 없이 LLM이 직접 답변을 생성합니다.
    """
    logger.info("---NODE: generate_llm_node---")
    question = state.get("question", "")
    
    try:
        generation = llm_answer_chain.invoke({"question": question})
        return {"generation": generation}
    except Exception as e:
        logger.error(f"LLM 답변 생성 오류: {e}")
        return {"generation": "답변 생성에 실패했습니다.", "error": str(e)}

def web_search_node(state: GraphState) -> GraphState:
    """
    Naver API를 통해 웹 검색을 수행하고 결과를 문자열로 저장합니다.
    """
    logger.info("---NODE: web_search_node---")
    question = state.get("question", "")
    
    # 현재 날짜 포함 (검색 품질 향상)
    today = datetime.now().strftime("%Y년 %m월 %d일")
    search_query = f"{question} ({today} 기준)"
    
    client_id = os.environ.get("NAVER_CLIENT_ID", "")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET", "")
    
    if not client_id or not client_secret:
        return {"web_results": "", "error": "네이버 검색 API 키가 설정되지 않았습니다."}
    
    url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=3"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # 문자열로 변환하여 저장
        results_str = ""
        for item in data.get("items", []):
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            desc = item.get("description", "").replace("<b>", "").replace("</b>", "")
            results_str += f"제목: {title}\n요약: {desc}\n\n"
            
        if not results_str:
            return {"web_results": "", "error": "검색 결과가 없습니다."}
            
        return {"web_results": results_str, "error": None}
    except Exception as e:
        logger.error(f"웹 검색 API 오류: {e}")
        return {"web_results": "", "error": "웹 검색 중 오류가 발생했습니다."}

def generate_web_search_node(state: GraphState) -> GraphState:
    """
    문자열로 저장된 웹 검색 결과를 문맥으로 답변을 생성합니다.
    """
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
        logger.error(f"웹 결과 기반 생성 오류: {e}")
        return {"generation": "웹 결과를 요약하는데 실패했습니다.", "error": str(e)}

# ==========================================
# 6. Edges (조건부 라우팅)
# ==========================================
MAX_RETRIES = 2

def route_question_edge(state: GraphState) -> str:
    """초기 질문을 분석하여 어떤 흐름(Node)으로 갈지 결정합니다."""
    logger.info("---EDGE: route_question_edge---")
    question = state.get("question", "")
    try:
        result_str = router_chain.invoke({"question": question})
        result = parse_with_fallback(result_str, RouteQuery)
        route = result.datasource
    except Exception:
        route = "llm"
        
    logger.info(f"결정된 라우트: {route}")
    if route == "vectorstore":
        return "vectorstore"
    elif route == "web_search":
        return "web_search"
    else:
        return "llm"

def decide_to_generate_edge(state: GraphState) -> str:
    """문서 평가 노드 이후, 문서를 바탕으로 생성할지 재작성할지 결정합니다."""
    logger.info("---EDGE: decide_to_generate_edge---")
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    
    if documents:
        logger.info("-> 관련 문서 있음 (generate)")
        return "generate"
    elif retry_count < MAX_RETRIES:
        logger.info(f"-> 관련 문서 없음. 재시도({retry_count}/{MAX_RETRIES}) (transform_query)")
        return "transform_query"
    else:
        logger.warning("-> 관련 문서 없음. 최대 재시도 초과. 종료 (end)")
        return "end"

def grade_generation_edge(state: GraphState) -> str:
    """생성된 답변의 환각 여부와 적절성을 판단합니다."""
    logger.info("---EDGE: grade_generation_edge---")
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)
    
    # 생성 노드에서 미리 에러 메시지를 넘긴 경우 바로 종료
    if "참고할 문서가 없어" in generation or "답변 생성에 실패" in generation:
        return "end"

    docs_text = "\n".join([d.page_content for d in documents])
    
    try:
        # 1. 환각(Hallucination) 검증
        hallucination_str = hallucination_chain.invoke({"documents": docs_text, "generation": generation})
        hallucination_res = parse_with_fallback(hallucination_str, GradeHallucination)
        
        if hallucination_res.binary_score == "no":
            logger.warning("-> 환각 감지됨 (not supported)")
            if retry_count < MAX_RETRIES:
                return "not supported" # 문서 내에서 다시 답변 생성 시도
            else:
                logger.warning("-> 환각 감지되었으나 최대 재시도 초과. 종료 (end)")
                return "end"
            
        # 2. 적절성(Answer usefulness) 검증
        answer_str = answer_grader_chain.invoke({"question": state.get("original_question", ""), "generation": generation})
        answer_res = parse_with_fallback(answer_str, GradeAnswer)
        
        if answer_res.binary_score == "yes":
            logger.info("-> 답변 적절성 통과 (useful)")
            return "useful"
        else:
            logger.warning("-> 답변 부적절 (not useful)")
            if retry_count < MAX_RETRIES:
                return "not useful" # 질문을 다시 작성하여 아예 처음부터 검색
            else:
                logger.warning("-> 답변 부적절하나 최대 재시도 초과. 종료 (end)")
                return "end"
    except Exception as e:
        logger.error(f"평가 엣지 오류: {e}")
        return "end"

# ==========================================
# 7. Graph Build
# ==========================================
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("transform_query_node", transform_query_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_web_search_node", generate_web_search_node)
workflow.add_node("generate_llm_node", generate_llm_node)

# 라우팅 엣지 (시작점)
workflow.set_conditional_entry_point(
    route_question_edge,
    {
        "vectorstore": "retrieve_node",
        "web_search": "web_search_node",
        "llm": "generate_llm_node"
    }
)

# VectorStore 분기
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

# Web Search 분기
workflow.add_edge("web_search_node", "generate_web_search_node")
workflow.add_edge("generate_web_search_node", END)

# LLM 분기
workflow.add_edge("generate_llm_node", END)

# 그래프 컴파일
app = workflow.compile()

# Mermaid 그래프 출력 (파일 저장 및 콘솔 일부 출력)
try:
    mermaid_png = app.get_graph().draw_mermaid()
    with open("graph_mermaid.txt", "w", encoding="utf-8") as f:
        f.write(mermaid_png)
    logger.info("Mermaid 그래프 코드가 graph_mermaid.txt 파일에 성공적으로 저장되었습니다.")
except Exception as e:
    logger.warning(f"Mermaid 그래프 생성 실패: {e}")

# ==========================================
# 8. Main (CLI 실행부)
# ==========================================
import traceback
from langgraph.errors import GraphRecursionError

def main():
    # Windows 환경 출력 인코딩 강제 설정 (이모지 등 출력 시 cp949 에러 방지)
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

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
                
            # 새로운 초기 state 생성 (이전 질문의 state가 섞이지 않도록 보장)
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
            
            # recursion_limit을 15로 설정하여 그래프 호출
            config = {"recursion_limit": 15}
            
            logger.info("에이전트 추론 시작...")
            try:
                # invoke를 사용해 최종 상태를 한 번에 가져옵니다.
                final_state = app.invoke(initial_state, config=config)
                
                # 최종 generation 출력 (generation이 없으면 error 필드 확인)
                final_generation = final_state.get("generation")
                if not final_generation or final_generation.strip() == "":
                    final_generation = final_state.get("error", "생성된 답변이 없습니다.")
                    
                print(f"\n[에이전트 답변]\n{final_generation}")
                
            except GraphRecursionError:
                print("\n[오류] 재시도 횟수를 초과했거나 무한 루프가 발생했습니다 (GraphRecursionError).")
                
            except requests.exceptions.ConnectionError as e:
                # Ollama 연결 또는 네이버 API 통신 오류
                if "11434" in str(e):
                    print("\n[오류] Ollama 서비스에 연결할 수 없습니다. Ollama 프로그램이 실행 중인지 확인하세요.")
                else:
                    print(f"\n[오류] 네트워크 연결 오류 (API 통신 실패 등): {e}")
                    
            except requests.exceptions.HTTPError as e:
                err_str = str(e)
                if e.response is not None and e.response.status_code == 404 and "11434" in err_str:
                    print(f"\n[오류] Ollama 모델을 찾을 수 없습니다. 모델 이름({LLM_MODEL})이 정확한지, 'ollama pull'을 수행했는지 확인하세요.")
                elif e.response is not None and e.response.status_code == 401 and "naver" in err_str:
                    print("\n[오류] Naver API 인증 실패. .env 파일의 클라이언트 ID와 시크릿 키를 확인하세요.")
                else:
                    print(f"\n[오류] HTTP 통신 오류: {e}")
                    
            except ValidationError:
                print("\n[오류] 구조화 출력(JSON) 파싱 중 오류가 발생했습니다. 모델이 정해진 형식을 따르지 않았습니다.")
                
            except Exception as e:
                # ChromaDB 예외 및 기타 예외
                err_str = str(e).lower()
                if "chroma" in err_str or "embed" in err_str:
                    print(f"\n[오류] VectorStore(Chroma) 또는 임베딩 모델(nomic-embed-text) 관련 오류가 발생했습니다: {e}")
                else:
                    print(f"\n[오류] 실행 중 알 수 없는 예외가 발생했습니다: {e}")
                    
                if log_level == "DEBUG":
                    traceback.print_exc()

        except KeyboardInterrupt:
            # Ctrl+C 입력 시 정상 종료
            print("\n\n입력이 취소되었습니다. 에이전트를 안전하게 종료합니다.")
            break

if __name__ == "__main__":
    main()
