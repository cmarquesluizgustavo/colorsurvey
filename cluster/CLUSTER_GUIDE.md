# Ambiente HTCondor

Referência do ambiente do cluster. O fluxo de trabalho do projeto está em `README.md`.

## Comandos úteis

```bash
condor_q -submitter $USER     # seus jobs na fila
condor_q -held                # jobs parados por erro
condor_q -analyze <job_id>    # por que um job não casa com nenhum slot
condor_status                 # slots e memória livre por nó
condor_history -limit 10      # jobs já concluídos
condor_rm <cluster_id>        # cancela uma submissão
condor_rm $USER               # cancela todos os seus jobs
```

## Jobs em *hold*

Duas causas comuns:

**Falta de memória.** A memória é pedida por job, na segunda coluna de
`cluster/experiments.txt`. Para mudar, ajuste `MEMORY_MAP` em
`cluster/generate_experiments_txt.py` e gere a lista de novo. Os nós têm tiers de
memória por slot diferentes, então um pedido maior reduz os slots elegíveis e o job
espera mais na fila.

**Falha na transferência de saída.** O HTCondor não cria diretórios intermediários no
nó de submissão: se o destino de um `transfer_output_remaps` não existir, a
transferência falha, o job vai para *hold* e a saída é perdida. O log do job mostra
`SHADOW ... failed to write to file ... (errno 2)`.

## Submeter apenas um subconjunto

Edite `cluster/experiments.txt` e comente (com `#`) ou remova as linhas que não quer
executar. Todas as linhas devem pertencer à mesma rodada.

## GPU

Um dos nós tem GPU, mas o driver é antigo e o PyTorch instalado cai para CPU
(`CUDA initialization: The NVIDIA driver on your system is too old`). Os experimentos
rodam em CPU; o aviso em `.err` é esperado.
