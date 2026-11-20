# M1-P2 — Conferma indipendente su 16 compiti (20 novembre 2026)

**Protocollo preregistrato prima della campagna:** [docs/PLASTICITY_CONFIRMATION.md](../docs/PLASTICITY_CONFIRMATION.md)  
**Esecuzione:** [GitHub Actions #44819212280](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44819212280)  
**Stato:** completata con successo, test automatici superati.  
**Dataset:** artefatto `plasticity-m1p2-24-seeds` (ID 14670766159), comprendente 1.536 righe di osservazioni, summary JSON e SVG.

## Risultato principale: nessuna evidenza robusta di deterioramento cumulativo

Confrontando la **stessa combinazione di famiglie di compito** in fasi diverse della vita delle reti, misuriamo il gap appaiato `AUC_continual − AUC_fresh`.

| Periodo | Compiti | Gap medio verso rete nuova |
|---|---|---:|
| Precoce, dopo la prima fase di apprendimento | 4–7 | **−2,65 punti percentuali** |
| Tardivo, dopo più compiti | 12–15 | **−2,95 punti percentuali** |
| **Differenza tardivo − precoce** | confronto appaiato per seed | **−0,30 pp** |

Intervallo bootstrap percentile descrittivo 95% per la differenza: **circa −2,30 / +1,68 pp** (24 seed, ricampionati a livello di sequenza). Su 13 seed il gap è diventato più negativo, su 11 è diventato meno negativo.

**Conclusione preregistrata:** il test **non conferma una progressiva riduzione della capacità di apprendere** attraverso queste 16 esperienze. Osserviamo ancora uno svantaggio della rete addestrata rispetto a una rete nuova, ma può trattarsi soprattutto di *negative transfer* o interferenza tra le rappresentazioni apprese, non necessariamente di deterioramento cumulativo. Questo è un risultato negativo utile.

## Risultato secondario: il rinnovo delle unità poco attive continua ad aiutare

In un nuovo insieme indipendente di seed (300–323), nei compiti **8–15** il rinnovo selettivo supera il rinnovo casuale di **+1,27 pp** di AUC media, intervallo bootstrap 95% circa **+0,59 / +1,90 pp**. L'effetto è positivo in **19 sequenze su 24**, negativo in 5.

Questo replica qualitativamente la campagna precedente, nella quale il vantaggio del rinnovo selettivo sul casuale era **+0,89 pp**.

## Metriche globali sui nuovi seed (compiti 1–15)

| Trattamento | AUC media apprendimento | Accuratezza finale media | Dimenticanza media |
|---|---:| ---:|---:|
| Continua | **71,83%** | 83,00% | 30,60 pp |
| Nuova a ogni compito | **75,57%** | 84,47% | 34,40 pp* |
| Rinnovo unità poco attive | **74,49%** | **85,24%** | 33,16 pp |
| Rinnovo casuale | **73,26%** | 83,66% | 32,25 pp |

\*La dimenticanza della rete nuova è intrinseca al reset totale e non va confrontata direttamente con una rete che conserva pesi.

Il rinnovo selettivo migliora sia la AUC sia l'accuratezza finale rispetto alla rete continua, ma peggiora un poco la conservazione dei compiti già appresi. Non è una vittoria senza costi.

## Controlli necessari

1. **Due campagne distinte:** M1-P ha usato seed 100–123 e 8 compiti; M1-P2 seed 300–323 e 16 compiti. M1-P2 è stato preregistrato dopo aver visto M1-P; è una conferma indipendente dei dati, non una prova dell'intero campo di applicazione.
2. **Metrica corretta:** AUC = area sotto la curva dell'accuratezza durante l'apprendimento di un singolo compito. Non è ROC AUC.
3. **Unità indipendente:** il seed/seq. di compiti, non ogni esempio di test né ogni compito osservato.
4. **Task sintetici e supervisionati:** classificazioni lineari, XOR, cerchio, strisce. Nessuna autonomia, coscienza, generazione di obiettivi né capacità di riscrivere il codice.
5. **Rinnovo esterno:** la rete non decide quando o come sostituire le unità. Il nostro algoritmo sceglie unità a bassa attivazione dai dati del training precedente.
6. **Nessun blocco generale della plasticità:** le unità completamente inattive restano rare. La causa delle difficoltà di transfer non è stata identificata causalmente.

## Decisione

**Non affermiamo la comparsa di una perdita cumulativa di plasticità.** La differenza precoce/tardiva è troppo piccola e incerta per farlo.

**Il vantaggio ripetuto del rinnovo selettivo, invece, merita un esperimento nuovo.** Il passo successivo dovrebbe rimuovere la regola esterna fissa del reset e far apprendere al sistema *quando* rinnovare le unità, con un costo esplicito di rinnovo e controlli identici. Questa sarebbe una prima prova di **meta-adattamento vincolato**, non di auto-riscrittura arbitraria.

Preregistrare prima:
- baseline nessun reset, reset casuale e reset selettivo fisso;
- controller che decide **se** e **quante** unità rinnovare, entro limiti stretti;
- costo computazionale esplicito e penalità per perdita di memoria;
- nuovi seed/famiglie di compiti mai utilizzati per scegliere il controller.

**Non modificare retroattivamente M1-P o M1-P2.** Le evidenze negative devono rimanere pubbliche.

## Dati e riproducibilità

[Workflow della campagna M1-P2](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44819212280), artefatto `plasticity-m1p2-24-seeds`:
- `measurements.csv` — tutte le 1.536 osservazioni;
- `summary.json` — statistiche aggregate e intervalli;
- `curves.svg` — AUC media per indice di compito.

Gli artefatti Actions hanno conservazione limitata a 30 giorni; il rapporto e i numeri principali sono archiviati stabilmente su GitHub.
