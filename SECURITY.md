# Security policy and API-key handling

## Critical: remove exposed API keys before publishing

Never publish an OpenAI API key or any other credential in code, notebook cells, outputs, screenshots, or Git history. **If a key has ever been exposed**, revoke it immediately, issue a replacement, and remove it from all distributed artifacts. Simply deleting a key from the latest commit does not erase older commits.

### Recommended local setup

```bash
export OPENAI_API_KEY='your-new-key'      # Linux/macOS shell, not in a tracked file
```

For Google Colab, use **Secrets**, make the secret accessible to the notebook, and load it from the Secrets manager or securely set an environment variable at runtime.

- Never commit `.env`, API keys, notebook runtime metadata containing secrets, or real patient data.
- Restrict API key usage, monitor usage costs, and rotate leaked keys.
- Review all proposed public notebook outputs before publication.
- If a key was committed, use an appropriate history-rewrite procedure (for example, `git filter-repo`) and coordinate the rewrite with collaborators.

## Clinical safety

Research outputs are generated for synthetic clinical profiles. Do not use them for medical diagnosis, patient-specific treatment, prescribing or clinical decision support.
