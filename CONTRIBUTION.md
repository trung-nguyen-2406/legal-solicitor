# Development & Contribution Guidelines

## 1. Branching Strategy

We follow a simplified **Trunk-Based / GitHub Flow** model:

* **`main`**: Production-ready branch. Must always pass all checks and be deployable. Direct commits to `main` are strictly forbidden.
* **Feature Branches**: Branch off from `main` using the following naming convention:
  * `feat/<feature-name>`: New capabilities or functional modules (e.g., `feat/state-machine-chunking`, `feat/qdrant-payload-indexing`).
  * `fix/<bug-name>`: Bug fixes and logic corrections (e.g., `fix/reranker-threshold-logic`).
  * `refactor/<target>`: Code restructuring without functional changes (e.g., `refactor/langgraph-orchestrator`).
  * `perf/<optimization>`: Performance improvements (e.g., `perf/onnx-runtime-batching`).
  * `chore/<task>`: Dependency updates, CI/CD configuration, or tooling (e.g., `chore/update-requirements`).
  * `docs/<topic>`: Documentation updates (e.g., `docs/update-sdd-v3`).

---

## 2. Commit Message Convention

All commit messages must adhere to the **Conventional Commits v1.0.0** specification:

```text
<type>(<scope>): <short description in present tense, lowercase, no period at end>

[optional body: explain the motivation and reasoning behind the change]

[optional footer: issue references, breaking changes]
```

### Commit Types

* `feat`: A new feature or capability.
* `fix`: A bug fix.
* `docs`: Documentation changes only.
* `refactor`: A code change that neither fixes a bug nor adds a feature.
* `perf`: A code change that improves performance or latency.
* `test`: Adding missing tests or correcting existing tests.
* `chore`: Maintenance tasks, build processes, or auxiliary tool changes.

### Examples

```text
feat(parser): implement lead-in preservation for legal article chunking
fix(retrieval): enforce active status pre-filter in qdrant hybrid query
docs(sdd): add deployment diagram and redis keyspace specifications
```

---

## 3. Pull Request & Merge Policy

* **Squash and Merge**: Always use **Squash and merge** when integrating feature branches into `main`. This maintains a linear, clean history on `main` where each commit corresponds to one complete deliverable.
* **Branch Cleanup**: Delete local and remote feature branches immediately after merging.

---

## 4. Code Standards & Environment

* **Formatting & Linting**: Python code must conform to PEP 8. Format using `black` and lint with `ruff` prior to pushing.
* **Secret Management**:
  * Never commit `.env` files, API keys, database credentials, or tokens.
  * Always update `.env.example` when introducing new environment variables.
* **Large Files**: Do not commit model weights, vector index checkpoints, or raw database dumps (`*.sqlite`, `*.db`). Store datasets in designated cloud storage or external drives.

---

## 5. Definition of Done (DoD)

A task or feature branch is considered **Done** and ready to merge into `main` only when all of the following criteria are met:

* [ ] **Code Cleanliness**: Debug code, commented-out blocks, and temporary print statements have been completely removed.
* [ ] **Error Handling**: Network operations and external service calls (Qdrant Cloud, Neon PostgreSQL, Upstash Redis, local Ollama) include proper timeouts and exception handling.
* [ ] **Local Verification**: The complete pipeline passes both the primary path (Happy Path) and relevant boundary edge cases (e.g., expired legal documents, empty retrieval responses).
* [ ] **Schema Synchronization**: Any modifications to database tables or Qdrant collection payloads are reflected in `init_databases.py` and documented in the SDD.
* [ ] **Configuration Updates**: All newly introduced environment variables are documented with sample values in `.env.example`.
* [ ] **Commit Integrity**: Commit messages follow the Conventional Commits format, and the branch is squashed cleanly into `main`.