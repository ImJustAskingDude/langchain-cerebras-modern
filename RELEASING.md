# Releasing `langchain-cerebras-modern`

This fork is prepared to publish as a drop-in replacement distribution with the
same Python import path:

```python
from langchain_cerebras import ChatCerebras
```

The distribution name on PyPI is:

```text
langchain-cerebras-modern
```

## Before first publish

1. Create the PyPI project `langchain-cerebras-modern`.
2. In GitHub, create two repository environments:
   - `testpypi`
   - `pypi`
3. In TestPyPI, add a trusted publisher for:
   - owner: `ImJustAskingDude`
   - repository: `langchain-cerebras-modern`
   - workflow: `.github/workflows/_test_release.yml`
   - environment: `testpypi`
4. In PyPI, add a trusted publisher for:
   - owner: `ImJustAskingDude`
   - repository: `langchain-cerebras-modern`
   - workflow: `.github/workflows/_release.yml`
   - environment: `pypi`
5. In GitHub repository secrets, add `CEREBRAS_API_KEY` so the release workflow can run integration tests.

## Local release commands

From `libs/cerebras`:

```bash
uvx --from poetry poetry build
uvx --from poetry poetry install --with test
uvx --from poetry poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests
```

## GitHub release flow

The repository already includes release workflows that:

1. build the package
2. publish to TestPyPI
3. run pre-release checks
4. publish to PyPI
5. create a GitHub release

The first manual run should be:

1. GitHub Actions -> `release`
2. branch: `main`
3. `working-directory`: `libs/cerebras`

If you prefer, you can add environment protection rules so only approved users
can deploy to `testpypi` and `pypi`.

## Integration test model selection

The live integration suite defaults to `zai-glm-4.7`. You can override the
models used in CI with GitHub Actions repository variables:

- `CEREBRAS_TEST_STANDARD_MODEL` for the provider-agnostic LangChain suite
- `CEREBRAS_TEST_ZAI_MODEL` for ZAI reasoning tests
- `CEREBRAS_TEST_NON_REASONING_MODEL` for plain-text smoke tests
- `CEREBRAS_TEST_GPT_OSS_MODEL` to enable GPT-OSS-specific reasoning tests

If `CEREBRAS_TEST_GPT_OSS_MODEL` is unset, the GPT-OSS-only tests are skipped.
