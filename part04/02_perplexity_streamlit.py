"""
실행방법 : c:\workspaces\venv\Scripts\python.exe -m streamlit run 02_perplexity_streamlit.py
웹브라우저에서 http://localhost:8501/ 접속
=======================================================================
[학습 교재용 소스 코드 설명]
이 파일은 LangGraph와 LangChain을 활용하여 'Perplexity' 스타일의 AI 검색 챗봇을 구현한 예제입니다.

주요 기능:
1. 사용자가 질문과 포커스(Focus: web, academic, video, math)를 입력합니다.
2. 챗봇 에이전트(LLM)가 질문을 분석하고, 선택된 포커스에 맞는 도구(Tool)를 호출합니다.
3. 도구가 외부 API(웹 검색, 논문 검색, 유튜브 스크립트 추출, 수학 계산 등)를 통해 정보를 수집합니다.
4. 수집된 정보를 바탕으로 최종 답변을 생성하여 사용자에게 제공합니다.

학습 포인트:
- LangChain의 `@tool` 데코레이터를 이용한 커스텀 도구 제작 방법
- 유튜브 영상을 검색하고 자막(Transcript)을 추출해 벡터 DB(Chroma)에 담고 RAG를 수행하는 방법
- LangGraph를 활용한 상태(State) 기반의 노드(Node) 및 엣지(Edge) 연결, 조건부 라우팅(Conditional Routing) 설정
=======================================================================
"""

# 필요한 라이브러리 및 모듈 임포트
import os
import ast
import warnings
from typing import TypedDict, Literal, Annotated

import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

# LangGraph 관련 모듈
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode

# LangChain 관련 모듈
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 외부 API 도구 래퍼 (논문 검색, 수학 계산, 웹 검색, 유튜브 로더 등)
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import YoutubeLoader
from youtube_search import YoutubeSearch

# 경고 메시지 무시 설정 (콘솔을 깔끔하게 유지하기 위함)
warnings.filterwarnings("ignore")

# 1. 환경 변수 로드 (.env 파일에서 API 키 등을 가져옴)
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 2. 비동기 루프 적용 (주피터 환경이나 특정 비동기 환경에서 충돌 방지)
# Streamlit 환경에서는 내부 서버 루프와 충돌(NoEventLoopError)을 일으키므로 비활성화합니다.
# nest_asyncio.apply()

# 3. 상태(State) 정의
# LangGraph에서 노드 간에 전달될 데이터 구조를 정의합니다.
class State(TypedDict):
    # 메시지 목록 (add_messages를 통해 기존 리스트에 새로운 메시지가 추가되는 방식)
    messages: Annotated[list, add_messages]
    # 사용자가 선택한 검색 포커스 영역
    focus: Literal["web", "academic", "video", "math"]
    
# =======================================================================
# [도구(Tools) 정의 파트]
# 각 포커스 영역에서 사용할 실제 검색 도구들을 정의합니다.
# =======================================================================

# 웹 검색 도구 (Tavily Search API 사용)
web_tool = TavilySearchResults(
    max_results=3,  # 최대 3개의 검색 결과 반환
    tavily_api_key=tavily_api_key
)

# 학술 논문 검색 도구 (Arxiv API 사용)
@tool
def academic_tool(query: str):
    """ Academic paper search tool """
    arxiv = ArxivAPIWrapper()
    docs = arxiv.run(query) # 검색어에 맞는 논문 정보 조회
    return docs

# 수학 계산 도구 (Wolfram Alpha API 사용)
@tool
def math_tool(query: str):
    """ Math tool """
    wolfram = WolframAlphaAPIWrapper()
    result = wolfram.run(query) # 수식이나 수학 질문 계산 결과 반환
    return result

# 비디오 검색 도구 (유튜브 검색 및 자막 기반 RAG 수행)
@tool
def video_tool(query: str):
    """
    Youtube 비디오를 조회하는 툴.
    질문을 분석해서 관련있는 비디오를 리턴한다.
    """
    # 1) 유튜브 검색을 통해 상위 3개의 영상 정보 가져오기
    results = YoutubeSearch(query, max_results=3).to_dict()
    # URL 주소 조립
    urls = [f"https://www.youtube.com{res['url_suffix']}" for res in results]
    
    docs = []
    # 2) 찾은 유튜브 URL들을 순회하며 영상 자막(스크립트) 추출
    for url in urls:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False, # pytube 오류를 방지하기 위해 메타데이터 조회 비활성화
            language=["en", "ko"], # 영어, 한국어 자막 시도
        )
        script = loader.load() # 자막 로드
        
        # 자막이 존재하는 경우 Document 객체로 생성하여 리스트에 추가
        if script:
            script_content = script[0].page_content
            title = script[0].metadata.get('title', 'Unknown Title')
            author = script[0].metadata.get('author', 'Unknown Author')
            
            doc = Document(
                page_content=script_content, 
                metadata={"source": url, "title": title, "author": author}
            )
            docs.append(doc)
            
    # 3) 추출한 모든 비디오 자막 내용을 벡터 DB에 저장 (RAG 처리)
    # 긴 스크립트를 1000자 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    
    # 임베딩 모델 (Nomic 텍스트 임베딩)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 분할된 텍스트를 메모리 기반 Chroma 벡터 DB에 저장
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        collection_name="video-db"
    )

    # 4) 사용자의 질문(query)과 가장 관련성이 높은 자막 내용(Relevant Docs) 검색
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)

    video_results = []
    # 5) 검색된 문서를 보기 좋게 문자열 형태로 포맷팅
    for doc in relevant_docs:
        title = doc.metadata.get("title", "Unknown Title")
        author = doc.metadata.get("author", "Unknown Author")
        script_content = doc.page_content

        video_info = f"""
        제목: {title}
        채널: {author}
        스크립트: 
        {script_content}
        """
        video_results.append(video_info)
    
    # 모든 비디오 검색 결과를 하나의 문자열로 결합하여 LLM에게 반환
    all_video_info = "\n\n".join(video_results)
    return all_video_info

# 각 포커스별로 호출할 도구들을 매핑
tools = {
    "web": [web_tool],
    "academic": [academic_tool],
    "video": [video_tool],
    "math": [math_tool]
}

# LangGraph에서 사용할 수 있도록 딕셔너리 형태의 ToolNode 객체로 변환
tool_nodes = {focus: ToolNode(tools[focus]) for focus in tools}

# =======================================================================
# [LLM 및 챗봇 로직 정의 파트]
# =======================================================================

# 사용할 LLM 모델 설정 (로컬 Ollama 활용, 답변의 일관성을 위해 temperature는 0으로 설정)
llm = ChatOllama(model="gemma4:e2b-it-qat", temperature=0)

# 챗봇 노드 함수: 상태(State)를 입력받아 LLM을 호출하고 결과를 반환
def chatbot(state: State):
    # 현재 포커스에 맞는 도구만 LLM에 바인딩(연결)
    llm_with_tools = llm.bind_tools(tools[state['focus']])
    # 도구가 연결된 LLM에 이전 메시지 기록을 전달하여 실행
    response = llm_with_tools.invoke(state['messages'])
    # 생성된 응답을 메시지 리스트에 추가하도록 반환
    return {"messages": [response]}

# =======================================================================
# [LangGraph 구축 파트]
# 노드(Node)와 엣지(Edge)를 정의하고 조건부 연결을 구성합니다.
# =======================================================================
graph_builder = StateGraph(State)

# 1. 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

# 2. 도구 노드 추가 (반복문을 통해 web_tools, academic_tools 등을 각각 노드로 추가)
for focus, tool_node in tool_nodes.items():
    graph_builder.add_node(f"{focus}_tools", tool_node)

# 조건부 라우팅 함수: LLM이 도구 사용을 요청했는지 여부를 판단
def tools_condition(state: State):
    # 마지막 메시지에 tool_calls(도구 호출 요청)이 포함되어 있다면
    if state['messages'][-1].tool_calls:
        # 해당 포커스의 도구 노드로 분기 이동
        return f"{state['focus']}_tools"
    # 도구 호출이 필요 없다면 대화(그래프) 종료
    return END

# 3. 조건부 엣지(Conditional Edges) 추가
graph_builder.add_conditional_edges(
    "chatbot",         # 시작 노드
    tools_condition,   # 라우팅 기준이 되는 함수
    {
        # 함수 결과값에 따라 이동할 다음 노드 매핑
        "web_tools": "web_tools",
        "academic_tools": "academic_tools",
        "video_tools": "video_tools",
        "math_tools": "math_tools",
        END: END
    }
)
    
# 4. 초기 엣지 설정 및 도구 실행 후 돌아올 엣지 설정
graph_builder.add_edge(START, "chatbot") # 그래프 시작 시 무조건 chatbot 노드로 진입

# 도구 노드가 실행을 마친 뒤에는 다시 chatbot 노드로 돌아와 도구 결과값을 판단하게 함
for focus in tools:
    graph_builder.add_edge(f"{focus}_tools", "chatbot")

# 5. 그래프 컴파일 (구조를 완성하고 실행 가능한 애플리케이션으로 빌드)
graph = graph_builder.compile()

# =======================================================================
# [Streamlit 기반 웹 UI 구현 파트]
# Streamlit은 프론트엔드 코드(HTML, CSS, JS) 없이
# 파이썬 스크립트만으로 웹 애플리케이션을 빠르게 만들 수 있게 해주는 라이브러리입니다.
# =======================================================================
# git 변경사항을 테스트 하기 위한 임시 주석


# st.title: 웹 브라우저 화면의 가장 위에 가장 큰 제목
# (h1 태그)을 출력합니다.
st.title("Perplexity 스타일 AI 검색 챗봇")

# 1. 세션 상태(Session State) 초기화
# st.session_state는 Streamlit 앱이 재실행(Rerun)되더라도 
# 사용자 데이터(채팅 기록 등)가 초기화되지 않고 유지되게 하는 전역 저장소입니다.
if "messages" not in st.session_state:
    st.session_state.messages = [] # 대화 기록을 담을 빈 리스트를 생성

# 2. 검색 포커스 선택 UI 생성
# st.header: 중간 크기의 제목(h2 태그)을 출력합니다.
st.header("검색 포커스 (Focus)")

# st.checkbox: 사용자가 체크할 수 있는 박스를 생성합니다. 체크 여부에 따라 True/False를 반환합니다.
focus_areas = {
    "web": st.checkbox("Web Search (웹 검색)"),
    "academic": st.checkbox("Academic Research (학술 논문)"),
    "video": st.checkbox("Video Search (유튜브 영상)"),
    "math": st.checkbox("Math (수학 계산)"),
}

selected_focus = []
# 딕셔너리를 순회하면서 체크된 항목(True)만 selected_focus 리스트에 추가합니다.
for area, key in focus_areas.items():
    if key: 
        selected_focus.append(area)

# 만약 사용자가 아무것도 선택하지 않았다면, 기본값으로 웹 검색('web')을 할당합니다.
if not selected_focus:
    selected_focus = ['web']

# 3. 기존 대화 기록 화면에 출력하기
# 세션 상태에 저장된 과거 채팅 기록을 순회하며 화면에 렌더링합니다.
for msg in st.session_state.messages:
    # st.chat_message: 'user'나 'assistant' 같은 역할(role)에 맞춰 프로필 아이콘과 말풍선을 만들어주는 컨테이너입니다.
    # with 블록 안에 작성된 내용이 해당 말풍선 안에 들어갑니다.
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"]) # 마크다운 포맷(볼드, 링크 등)을 지원하는 텍스트 출력 함수

# 4. 사용자 입력 받기 및 처리
# st.chat_input: 하단에 채팅을 칠 수 있는 입력창을 만듭니다. 
# 왈러스 연산자(:=)를 사용해, 사용자가 엔터를 쳤을 때 그 문자열을 prompt 변수에 바로 할당합니다.
if prompt := st.chat_input("질문을 입력하세요..."):
    
    if not selected_focus:
        # st.error: 빨간색 에러 메시지 박스를 출력합니다.
        st.error("적어도 하나의 검색 영역(Focus)을 선택해야 합니다.")
    else:
        # [사용자 입력 처리]
        # 입력한 텍스트를 세션 상태(기록)에 딕셔너리 형태로 저장합니다.
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 사용자의 질문을 곧바로 화면에 말풍선으로 띄워줍니다.
        with st.chat_message("user"):
            st.markdown(prompt)

        # [AI 처리 과정 표시]
        # st.status: 작업 진행 상황을 접었다 펼 수 있는 상자(Status container) 형태로 시각화합니다.
        # expanded=True이면 상자가 펼쳐져서 안의 내용이 보입니다.
        with st.status("요청을 처리하는 중입니다...", expanded=True) as status:
            responses = []
            
            # 사용자가 다중 선택한 포커스별로 순회하며 LangGraph를 호출합니다.
            for focus in selected_focus:
                # status.write: 상태 컨테이너 내부에 텍스트를 출력합니다.
                status.write(f"{focus.capitalize()} 검색 진행 중...")

                # 사전에 빌드해둔 LangGraph(graph)에 현재까지의 모든 대화 기록과 포커스를 전달해 실행시킵니다.
                result = graph.invoke({
                    "messages": st.session_state.messages,
                    "focus": focus
                })

                # 그래프 실행 결과의 마지막 메시지(AI의 최종 답변)를 추출합니다.
                assistant_response = result["messages"][-1].content
                # (Focus종류, AI답변) 형태의 튜플로 리스트에 저장합니다.
                responses.append((focus, assistant_response))
                
                status.write(f"{focus.capitalize()} 검색 완료!")
            
            # 모든 처리가 끝나면 상태 박스의 제목을 바꾸고 상자를 접습니다(expanded=False).
            status.update(label="처리가 완료되었습니다!", state="complete", expanded=False)

    # [최종 결과 출력]
    # 각 포커스별로 수집된 AI의 응답을 화면에 표시하고 세션 기록에 추가합니다.
    for focus, response in responses:
        # 응답 텍스트 앞에 어떤 포커스의 답변인지 볼드체(**)로 덧붙입니다.
        response_with_focus = f"**({focus.upper()})**\n{response}"
        
        # 챗봇의 답변을 세션(기록)에 저장합니다.
        st.session_state.messages.append({"role": "assistant", "content": response_with_focus})
        
        # 챗봇의 프로필 아이콘과 함께 답변을 화면에 마크다운으로 렌더링합니다.
        st.chat_message("assistant").markdown(response_with_focus)

# 5. 대화 초기화 버튼 기능
# st.button: 클릭할 수 있는 버튼을 생성합니다. 클릭 시 True를 반환합니다.
if st.button("Clear Chat (대화 초기화)"):
    # 세션 상태의 메시지 리스트를 빈 리스트로 덮어씌워 초기화합니다.
    st.session_state.messages = []
    # st.rerun(): 스크립트를 강제로 처음부터 다시 실행시켜 초기화된 화면을 새로고침(반영)합니다.
    st.rerun()
