# Kaggle Tokens

Create these folders and put one `kaggle.json` in each:

```text
.kaggle_tokens/account1/kaggle.json
.kaggle_tokens/account2/kaggle.json
.kaggle_tokens/account3/kaggle.json
```

Get `kaggle.json` from:

```text
Kaggle -> Settings -> API -> Create New Token
```

Never commit real `kaggle.json` tokens.

New Kaggle CLI versions also accept an `access_token` text file. This repo's orchestrator supports both:

```text
.kaggle_tokens/account1/kaggle.json
.kaggle_tokens/account1/access_token
```
