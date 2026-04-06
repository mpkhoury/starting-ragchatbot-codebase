# Como uma pergunta do usuário é processada

```mermaid
flowchart TD
    subgraph FE["🖥️  FRONTEND  (browser)"]
        A(["👤 Usuário digita uma pergunta"])
        B["script.js envia\nPOST /api/query\n{ query, session_id }"]
        Z(["💬 Resposta exibida no chat\ncom fontes"])
    end

    subgraph BE["⚙️  BACKEND  (FastAPI — app.py)"]
        C["Recebe a requisição\nCria sessão se necessária"]
    end

    subgraph RAG["🔗  ORQUESTRADOR  (rag_system.py)"]
        D["Busca histórico\nda conversa"]
        E["Monta o prompt\ncom contexto"]
    end

    subgraph AI["🤖  CAMADA DE IA  (ai_generator.py)"]
        F["Envia para Claude API\njunto com ferramentas disponíveis"]
        G{{"Claude decide:\npreciso buscar\nnos documentos?"}}
        H["Claude recebe os resultados\ne gera a resposta final"]
    end

    subgraph TOOL["🔧  FERRAMENTA DE BUSCA  (search_tools.py)"]
        I["Executa search_course_content\ncom query + filtros do Claude"]
    end

    subgraph DATA["📦  DADOS  (vector_store.py + ChromaDB)"]
        J["Converte a query\nem embedding numérico"]
        K[("ChromaDB\nencontra os 5 chunks\nmais similares")]
        L["Retorna trechos\ncom metadados\n(curso, lição)"]
    end

    %% Flow
    A --> B --> C --> D --> E --> F --> G

    G -- "Sim" --> I
    I --> J --> K --> L --> I
    I -- "resultados formatados" --> H

    G -- "Não\n(resposta direta)" --> H

    H --> BE
    BE --> Z

    %% Styling
    style FE fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style BE fill:#dcfce7,stroke:#22c55e,color:#14532d
    style RAG fill:#fef9c3,stroke:#eab308,color:#713f12
    style AI fill:#fae8ff,stroke:#a855f7,color:#581c87
    style TOOL fill:#ffedd5,stroke:#f97316,color:#7c2d12
    style DATA fill:#f1f5f9,stroke:#64748b,color:#1e293b

    style A fill:#3b82f6,color:#fff,stroke:#1d4ed8
    style Z fill:#3b82f6,color:#fff,stroke:#1d4ed8
    style G fill:#a855f7,color:#fff,stroke:#7e22ce
    style K fill:#475569,color:#fff,stroke:#334155
```

---

## Passo a passo

| # | O que acontece | Arquivo |
|---|---|---|
| 1 | Usuário digita e envia a pergunta | `frontend/script.js` |
| 2 | Frontend faz `POST /api/query` com a pergunta e o ID da sessão | `frontend/script.js` |
| 3 | Backend recebe, cria sessão se necessário | `backend/app.py` |
| 4 | Orquestrador recupera histórico da conversa | `backend/session_manager.py` |
| 5 | Monta o prompt com o histórico e passa ao Claude | `backend/rag_system.py` |
| 6 | Claude decide se precisa buscar nos documentos | `backend/ai_generator.py` |
| 7 | **Se sim:** executa a ferramenta `search_course_content` | `backend/search_tools.py` |
| 8 | A pergunta vira um vetor numérico (embedding) | `backend/vector_store.py` |
| 9 | ChromaDB encontra os 5 trechos mais relevantes | ChromaDB |
| 10 | Os trechos voltam ao Claude para ele formular a resposta | `backend/ai_generator.py` |
| 11 | Resposta + fontes são enviadas ao frontend | `backend/app.py` |
| 12 | Chat exibe a resposta com as referências | `frontend/script.js` |
