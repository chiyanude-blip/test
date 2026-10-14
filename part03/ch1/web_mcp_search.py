"""
=======================================================================
[LangGraph + LangChain + MCP 기반 웹 검색 프로그램]
=======================================================================
이 프로그램은 LangChain 및 LangGraph를 기반으로 Anthropic MCP(Model Context Protocol)
표준을 준수하는 외부 도구 서버(Tavily MCP)와 연동하여 실시간 외부 데이터를 검색하고
종합적인 답변을 제공하는 대화형 콘솔 애플리케이션입니다.

=======================================================================
[핵심 아키텍처 및 MCP(Model Context Protocol) 개념 정리]
=======================================================================
1. MCP(Model Context Protocol)란?
   - Anthropic이 제안한 개방형 프로토콜 표준으로, LLM(애플리케이션)과 외부 데이터/도구 서버 간의
     상호작용 규격을 표준화한 것입니다.
   - 데이터베이스, 파일 시스템, 검색 엔진(Tavily), 사내 API 등 다양한 도구를 일관된 규격(JSON-RPC 기반)으로
     손쉽게 플러그인처럼 연결할 수 있도록 지원합니다.

2. MCP 연동 방식 (Stdio vs SSE):
   - Stdio (표준 입출력 파이프 방식):
     * 클라이언트 애플리케이션이 자식 프로세스(Subprocess)로 MCP 서버 프로그램을 직접 실행하고,
       stdin/stdout 파이프를 통해 JSON-RPC 메시지를 주고받는 방식입니다.
     * 본 프로그램에서 사용하는 방식으로, 별도의 외부 호스팅 없이 로컬 환경(npx 등)에서 직접 구동할 때 적합합니다.
   - SSE (Server-Sent Events / HTTP 방식):
     * 원격 서버에서 실행 중인 독립적인 MCP 엔드포인트에 HTTP/SSE로 연결하는 방식입니다.

3. MultiServerMCPClient의 역할:
   - 다수의 MCP 서버(예: 웹 검색, 파일 시스템, 데이터베이스 등) 설정을 단일 클라이언트로 통합 관리합니다.
   - MCP 서버들이 제공하는 각 도구(Tool) 사양을 자동으로 감지(Discovery)하고, 이를 LangChain이 이해할 수 있는
     표준 도구 객체(BaseTool) 형태로 자동 변환 및 래핑해 줍니다.
=======================================================================
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Annotated, TypedDict, List, Dict, Any

from dotenv import load_dotenv

# -----------------------------------------------------------------------
# Windows 터미널 한글 및 특수기호/이모지 출력 인코딩 보정
# -----------------------------------------------------------------------
# Windows의 기본 콘솔 인코딩(cp949)으로 인해 발생하는 한글 깨짐 및 UnicodeEncodeError를 방지합니다.
if hasattr(sys.stdout, 'reconfigure') and sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =======================================================================
# 1. 환경 변수(.env) 다중 경로 자동 탐색 및 로드
# =======================================================================
# 작업 디렉터리가 다르더라도(프로젝트 루트, 하위 폴더 등) 유연하게 설정 파일을 찾을 수 있도록
# 현재 폴더부터 상위 폴더들을 순차적으로 탐색합니다.
current_dir = Path(__file__).resolve().parent
search_env_paths = [
    current_dir / ".env",
    current_dir.parent / ".env",
    current_dir.parent.parent / ".env",
    current_dir.parent.parent / "part04" / ".env",
]
for p in search_env_paths:
    if p.exists():
        load_dotenv(p)
        break

# Tavily 웹 검색 API 키 로드
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# =======================================================================
# 2. 필수 라이브러리 임포트 (LangChain, LangGraph, MCP 어댑터)
# =======================================================================
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# langchain_mcp_adapters: LangChain과 Model Context Protocol을 잇는 공식 브릿지 라이브러리
from langchain_mcp_adapters.client import MultiServerMCPClient


# =======================================================================
# 3. 확장 가능한 MCP 서버 레지스트리 설정 (심층 분석)
# =======================================================================
def get_mcp_server_configs() -> Dict[str, Any]:
    """
    등록할 MCP(Model Context Protocol) 서버 목록 및 연결 구성을 반환합니다.

    ---------------------------------------------------------------------
    [Stdio 전송 방식 설정 세부 해설]
    ---------------------------------------------------------------------
    1. 'transport': 'stdio'
       - 클라이언트 프로그램이 자식 프로세스로 서버 명령어를 구동하고 표준 입출력(stdio) 스트림을 통해
         JSON-RPC 2.0 프로토콜로 통신합니다.

    2. 'command':
       - Windows 환경에서는 CLI 명령어가 배치 파일(.cmd)로 제공되는 경우가 많습니다.
         예를 들어 Node.js 패키지 실행기인 npx는 Windows에서 `npx.cmd`로 호출해야 정확히 실행됩니다.
         (Linux/macOS는 `npx`)

    3. 'args':
       - 실행 명령어에 전달할 인자 배열입니다.
       - ["-y", "tavily-mcp"]: `-y` 플래그는 npx가 패키지를 사전에 전역 설치하지 않았더라도
         사용자에게 확인을 묻지 않고 즉시 다운로드하여 실행하도록 합니다.
       - "tavily-mcp"는 Tavily에서 공식 제공하는 표준 MCP 서버 패키지입니다.

    4. 'env':
       - 자식 프로세스에 전달할 환경 변수 딕셔너리입니다.
       - **중요**: `**os.environ`을 통해 부모 프로세스의 환경 변수(특히 PATH, SYSTEMROOT, TEMP 등)를
         반드시 상속해야 Node.js, npm, npx 실행 경로를 찾을 수 있습니다.
       - Tavily MCP 서버가 API 호출 시 인증할 `TAVILY_API_KEY`를 환경 변수로 주입합니다.
    ---------------------------------------------------------------------
    """
    is_windows = sys.platform == "win32"
    npx_cmd = "npx.cmd" if is_windows else "npx"

    configs = {
        # 1) Tavily 공식 웹 검색 MCP 서버 (Stdio 서브프로세스 방식)
        "tavily": {
            "transport": "stdio",
            "command": npx_cmd,
            "args": ["-y", "tavily-mcp"],
            "env": {
                **os.environ,  # Node.js 및 OS 기본 PATH 유지를 위해 필수 상속
                "TAVILY_API_KEY": TAVILY_API_KEY,
            },
        },
        # -------------------------------------------------------------
        # [다중 서버 확장 예시 1] 원격 Tavily SSE 엔드포인트 연동 시:
        # "tavily_remote": {
        #     "transport": "sse",
        #     "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        # },
        # -------------------------------------------------------------
        # [다중 서버 확장 예시 2] 로컬 파일 시스템 MCP 도구 추가 시:
        # "filesystem": {
        #     "transport": "stdio",
        #     "command": npx_cmd,
        #     "args": ["-y", "@modelcontextprotocol/server-filesystem", str(current_dir.parent.parent)],
        #     "env": {**os.environ},
        # },
        # -------------------------------------------------------------
        # [다중 서버 확장 예시 3] SQLite 데이터베이스 MCP 도구 추가 시:
        # "sqlite": {
        #     "transport": "stdio",
        #     "command": "uvx",
        #     "args": ["mcp-server-sqlite", "--db-path", "test.db"],
        # },
        # -------------------------------------------------------------
    }
    return configs


# =======================================================================
# 4. LLM 모델 초기화
# =======================================================================
# 로컬 Ollama를 기본 엔진으로 사용합니다.
# Ollama가 설치되어 있고 모델(gemma4:e2b-it-qat 등)이 pull되어 있어야 합니다.
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-it-qat")
llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)


# =======================================================================
# 5. LangGraph 상태(State) 정의
# =======================================================================
class AgentState(TypedDict):
    """
    LangGraph 워크플로우 전반에 공유되는 상태 컨텍스트 정의
    
    - messages: 대화 히스토리 목록
      Annotated[..., add_messages] 리듀서를 사용하여 새로운 메시지가 전달될 때마다
      기존 목록을 덮어쓰지 않고 뒤에 덧붙이는(append) 방식으로 상태를 유지합니다.
    """
    messages: Annotated[List[BaseMessage], add_messages]


# =======================================================================
# 6. LangGraph 워크플로우 빌더 함수 (그래프 구조 정의)
# =======================================================================
def build_search_graph(llm, tools, checkpointer=None):
    """
    주어진 LLM과 MCP 도구 목록을 기반으로 LangGraph 에이전트 그래프를 생성합니다.

    [동작 흐름]
    START -> chatbot -> (조건부 분기: tools_condition)
                          ├─> 도구 호출 필요: tools (ToolNode) -> 다시 chatbot으로 복귀 (루프)
                          └─> 최종 답변 완료: END
    """
    
    # 시스템 프롬프트 정의: 모델이 사용자 질의에 답할 때 도구를 사용하는 원칙과 언어 규정을 지시
    system_prompt = (
        "당신은 최신 외부 데이터를 검색하여 정확하고 유용한 정보를 제공하는 전문 AI 어시스턴트입니다.\n"
        "사용자의 질문에 답하기 위해 필요한 경우 언제든지 제공된 MCP 도구(tavily_search 등)를 적극적으로 활용하세요.\n"
        "검색된 결과를 종합하여 한국어로 명확하고 친절하게 답변해주세요."
    )

    # LLM에 MCP 도구들의 스키마(Tool Definition)를 바인딩
    # LLM은 이를 통해 자신이 언제 어떤 인자(arguments)로 도구를 호출해야 하는지 인지합니다.
    llm_with_tools = llm.bind_tools(tools)

    # 1. 챗봇(LLM 추론) 노드
    async def chatbot_node(state: AgentState):
        messages = state["messages"]
        # 시스템 프롬프트가 대화 첫 머리에 없는 경우 삽입
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        print(f"   💬 [DEBUG] Chat Node Input: {messages}")

        # LLM 비동기 추론 호출
        response = await llm_with_tools.ainvoke(messages)
        print(f"   💬 [DEBUG] Chat Node Output: {response}")

        # 반환값은 state['messages']에 추가됨
        return {"messages": [response]}

    # 2. 도구 실행 노드 (ToolNode)
    # LangGraph의 사전 빌드된 ToolNode는 LLM의 AIMessage에 포함된 tool_calls를 감지하고
    # 알맞은 MCP 도구를 실행한 뒤 그 결과를 ToolMessage 형태로 반환합니다.
    tool_node = ToolNode(tools)

    # 3. StateGraph 그래프 빌더 생성 및 노드 등록
    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)

    # 시작점 설정: 그래프가 실행되면 chatbot 노드부터 시작
    builder.add_edge(START, "chatbot")

    # 조건부 라우팅 (tools_condition):
    # - LLM의 응답에 tool_calls가 있으면 "tools" 노드로 이동
    # - tool_calls가 없으면 (답변 생성이 완료되었으면) END 노드로 이동하여 종료
    builder.add_conditional_edges("chatbot", tools_condition)

    # 도구 실행 후 처리: 도구 실행 결과(ToolMessage)를 가지고 다시 챗봇으로 복귀하여
    # 검색된 정보를 바탕으로 최종 사용자 답변을 작성하도록 루프를 구성
    builder.add_edge("tools", "chatbot")

    # checkpointer(MemorySaver)를 등록하여 멀티턴 대화 세션 상태를 저장할 수 있도록 컴파일
    return builder.compile(checkpointer=checkpointer)


# =======================================================================
# 7. 대화형 메인 실행 함수
# =======================================================================
async def main():
    print("=" * 65)
    print(" 🚀 LangGraph & Tavily MCP 확장형 웹 검색 시스템")
    print("=" * 65)

    # 필수 환경 변수 검증
    if not TAVILY_API_KEY:
        print("⚠️ [경고] TAVILY_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 TAVILY_API_KEY='tvly-...' 형태로 등록해주세요.")
        return

    # -------------------------------------------------------------------
    # [단계 1] MultiServerMCPClient 초기화 및 MCP 도구 동적 로드
    # -------------------------------------------------------------------
    # 1. get_mcp_server_configs()로 정의된 서버 설정 목록을 로드
    mcp_configs = get_mcp_server_configs()
    print(f"[*] MCP 서버 연결 구성: {list(mcp_configs.keys())}")
    print("[*] MCP 서버로부터 도구 목록 로드 중 (잠시만 기다려주세요)...")

    try:
        # MultiServerMCPClient 생성:
        # 설정에 지정된 command(npx tavily-mcp)를 서브프로세스로 실행하고 stdio 통신을 수립합니다.
        mcp_client = MultiServerMCPClient(mcp_configs)

        # get_tools():
        # 각 MCP 서버에 ListToolsRequest를 전송하여 도구 목록 및 JSON Schema를 질의하고,
        # 이를 LangChain 호환 Tool 객체 리스트로 변환하여 가져옵니다.
        tools = await mcp_client.get_tools()
    except Exception as e:
        print(f"❌ MCP 도구 로드 실패: {e}")
        return

    if not tools:
        print("❌ 사용 가능한 MCP 도구를 가져오지 못했습니다.")
        return

    # 로드된 도구 목록 및 설명 출력
    print(f"✅ 총 {len(tools)}개의 MCP 도구 로드 완료:")
    for t in tools:
        desc = t.description[:50] + "..." if len(t.description) > 50 else t.description
        print(f"   - {t.name}: {desc}")
    print("-" * 65)

    # -------------------------------------------------------------------
    # [단계 2] LangGraph 에이전트 빌드 (MemorySaver 체크포인터 활성화)
    # -------------------------------------------------------------------
    # MemorySaver는 세션 ID(thread_id)별로 대화 상태(State)를 메모리에 유지하여
    # 이전 대화 내용을 기억하는 멀티턴 대화를 가능하게 만듭니다.
    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "web_search_session"}}

    try:
        graph = build_search_graph(llm, tools, checkpointer=memory)
        print(f"✅ LangGraph 에이전트 준비 완료 (LLM: ChatOllama - {DEFAULT_MODEL})")
        print(f"✅ 대화 히스토리 메모리(MemorySaver) 활성화 완료")
    except Exception as e:
        print(f"❌ 그래프 빌드 실패: {e}")
        return

    print("\n💡 사용 안내:")
    print(" - 질문을 입력하면 Tavily MCP를 통해 실시간 웹 정보를 검색하여 답변합니다.")
    print(" - 멀티턴 대화가 지원되므로 이전 답변에 이어서 질문하실 수 있습니다.")
    print(" - 종료하려면 'q', 'quit', 'exit' 중 하나를 입력하세요.")
    print("=" * 65)

    # -------------------------------------------------------------------
    # [단계 3] 대화형 사용자 입력 루프
    # -------------------------------------------------------------------
    while True:
        try:
            user_input = input("\n[질문 입력] > ").strip()
            if not user_input:
                continue

            # 프로그램 종료 감지
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\n👋 프로그램을 종료합니다. 이용해 주셔서 감사합니다!")
                break

            print("\n🔍 검색 및 답변 생성 진행 중...")
            inputs = {"messages": [HumanMessage(content=user_input)]}

            # -----------------------------------------------------------
            # 그래프 실행 및 이벤트 스트리밍 (astream)
            # -----------------------------------------------------------
            # stream_mode="updates": 노드가 실행을 마칠 때마다 변경된 상태(state update)를 실시간 수신
            try:
                final_response = None
                async for event in graph.astream(inputs, config=thread_config, stream_mode="updates"):
                    for node_name, output in event.items():
                        # 도구 노드가 실행되었을 때
                        if node_name == "tools":
                            print("   🛠️  [MCP Tool] 외부 도구 실행 및 데이터 수신 완료")
                        
                        # 챗봇 노드가 실행되었을 때
                        elif node_name == "chatbot":
                            last_msg = output["messages"][-1]
                            print(f"   💬 [DEBUG] Chat Node Output: {output}")

                            # 도구 호출(Tool Calls) 요청이 포함되어 있는지 점검
                            if getattr(last_msg, "tool_calls", None):
                                for call in last_msg.tool_calls:
                                    query_arg = call.get('args', {}).get('query', '')
                                    print(f"   ⚙️  [MCP Call] 도구 호출 요청: {call['name']}(query='{query_arg}')")
                            else:
                                # 도구 호출이 없는 최종 텍스트 응답 저장
                                final_response = last_msg.content

                # 최종 응답 출력
                print("\n" + "=" * 50)
                print("📢 [최종 답변]")
                print("=" * 50)
                if final_response:
                    print(final_response)
                else:
                    print("답변을 생성하지 못했습니다.")
                print("=" * 50)

            except Exception as graph_err:
                print(f"\n⚠️ 그래프 실행 중 오류가 발생했습니다: {graph_err}")
                print("💡 (로컬 Ollama 서비스가 켜져 있는지 확인해주세요.)")
                
                # 비상 Fallback 모드:
                # LLM 연결 또는 추론 오류가 발생하더라도, 로드된 Tavily MCP 도구를 직접 호출하여
                # 원시 검색 결과를 사용자에게 표시해 주는 장애 대비 로직입니다.
                print("\n[비상 검색 모드] Tavily MCP 도구 직접 검색 결과:")
                search_tool = next((t for t in tools if "search" in t.name), None)
                if search_tool:
                    try:
                        raw_result = await search_tool.ainvoke({"query": user_input})
                        print(str(raw_result)[:1000] + ("..." if len(str(raw_result)) > 1000 else ""))
                    except Exception as tool_err:
                        print(f"직접 도구 호출 오류: {tool_err}")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 프로그램을 종료합니다.")
            break


# =======================================================================
# 프로그램 시작점 (Entry Point)
# =======================================================================
if __name__ == "__main__":
    try:
        # 비동기 메인 함수 실행
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
