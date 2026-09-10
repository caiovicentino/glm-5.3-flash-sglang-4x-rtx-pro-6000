#!/usr/bin/env bash
# Funções comuns das skills de operação da GPU.
# Uso:  source "$(git rev-parse --show-toplevel)/vast/lib.sh"
#
# Descobre a instância ativa sozinho — IP, portas e até o id mudam a cada
# aluguel, e hardcodar isso já nos deixou com o chat apontando para máquina
# morta mais de uma vez.

VAST_KEY="$(cat ~/.config/vastai/vast_api_key 2>/dev/null)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/secrets/env.sh" ] && source "$REPO/secrets/env.sh"

vast_api() {  # vast_api <caminho> [args extras do curl]
  curl -s -m 30 "https://console.vast.ai/api/v0/$1" \
    -H "Authorization: Bearer $VAST_KEY" "${@:2}"
}

# Preenche INST_ID, INST_IP, INST_API_PORT, INST_SSH_PORT, INST_DPH, INST_MACHINE,
# INST_STATUS, INST_START. Devolve 1 se não houver instância.
instancia_ativa() {
  local raw; raw="$(vast_api instances/)"
  eval "$(printf '%s' "$raw" | python3 -c "
import sys,json
ins=json.load(sys.stdin).get('instances',[])
if not ins: sys.exit(0)
i=max(ins,key=lambda x: x.get('start_date') or 0)   # a mais recente
p=(i.get('ports') or {})
def porta(k):
    v=p.get(k); return v[0]['HostPort'] if v else ''
print('INST_ID=%s'%i['id'])
print('INST_IP=%s'%(i.get('public_ipaddr') or ''))
print('INST_API_PORT=%s'%porta('8000/tcp'))
print('INST_SSH_PORT=%s'%porta('22/tcp'))
print('INST_DPH=%s'%(i.get('dph_total') or 0))
print('INST_MACHINE=%s'%(i.get('machine_id') or ''))
print('INST_STATUS=%s'%(i.get('actual_status') or ''))
print('INST_START=%s'%(i.get('start_date') or 0))
print('INST_PREPAGO=%s'%(i.get('credit_balance') or 0))   # saldo PRE-PAGO da instancia (reserva)
print('INST_DESCONTO=%s'%(i.get('credit_discount') or 0)) # desconto da reserva (0.025 = 2,5%)
")"
  [ -n "${INST_ID:-}" ] || return 1
  INST_BASE="http://$INST_IP:$INST_API_PORT"
}

# Executa comando na instância. StrictModes fica desligado pelo onstart, mas se
# o SSH falhar não trave a skill inteira — várias checagens são via HTTP.
# Multiplexação: a 1ª conexão abre um socket mestre e as seguintes o reaproveitam
# (~0,2 s em vez de 2–5 s de handshake). O painel faz até 6 chamadas por leitura;
# em 04/09 cada handshake passou a levar 4,6 s e o /api demorava 1m52 — o navegador
# desistia antes. ControlPersist mantém o mestre 10 min ocioso.
gpu_ssh() {
  ssh -p "$INST_SSH_PORT" "root@$INST_IP" \
      -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR \
      -o ControlMaster=auto -o ControlPath=/tmp/cb-ssh-%C -o ControlPersist=600 \
      "$@" 2>/dev/null
}

gpu_ssh_ok() { gpu_ssh 'echo ok' | grep -q ok; }

# Métrica agregada do /metrics do vLLM (soma labels).
metrica() {  # metrica <nome>
  curl -s -m 15 "$INST_BASE/metrics" 2>/dev/null | python3 -c "
import sys
alvo=sys.argv[1]; t=0.0; achou=False
for ln in sys.stdin:
    if ln.startswith('#'): continue
    nome=ln.split('{')[0] if '{' in ln else ln.rsplit(' ',1)[0]
    if nome.strip()==alvo:
        try: t+=float(ln.rsplit(' ',1)[1]); achou=True
        except: pass
print(t if achou else '')" "$1"
}

api_http() { curl -s -m 10 -o /dev/null -w '%{http_code}' \
  "$INST_BASE/v1/models" -H "Authorization: Bearer ${VLLM_API_KEY:-x}"; }

# Formata número grande com separador de milhar em pt-BR.
num() { python3 -c "print(f'{int(float(\"${1:-0}\")):,}'.replace(',','.'))" 2>/dev/null || echo "$1"; }
