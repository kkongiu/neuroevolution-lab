# M3 — Novità imprevedibile, compiti ricorrenti e controllo dello stato neurale

**Data:** 21 novembre 2026  
**Esito primario:** **NEGATIVO / vantaggio dell'autoregolazione non dimostrato**.  
**Esecuzione originale:** [GitHub Actions #44884705352](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44884705352), completata con successo  
**Codice congelato per la campagna:** commit dac462db45d0daffea180655e532b174936aff01  
**Protocollo precedentemente congelato:** [docs/M3_PROTOCOL.md](../docs/M3_PROTOCOL.md)  
**Pull request originale:** [#7](https://github.com/kkongiu/neuroevolution-lab/pull/7)

## Domanda preregistrata

In un ambiente con successione stocastica di compiti nuovi e ritorno di compiti già incontrati, una policy evoluta che osserva errori e attività della propria rete sceglie meglio *quando* rigenerare neuroni rispetto a una policy che vede **soltanto l'indice del compito**?

La variabile primaria era il vantaggio AUC tra controllore adattivo e controllore temporale, sulla distribuzione parzialmente **nuova** (shifted); ogni sequenza finale viene considerata una replica statistica.

## Disegno

- **Meta-training:** 4 seed 8000–8003; **meta-validation:** 3 seed 9000–9002.
- **Test finale regolare:** 12 seed 10000–10011, classi line, XOR, circle, stripe in ordine casuale.
- **Test finale con distribuzione parzialmente nuova:** 12 seed 11000–11011, classi line, checker, wedge, stripe; checker e wedge non presenti nel meta-training.
- **10 compiti per sequenza**. I primi due sono nuovi; da lì ogni nuovo passaggio ha probabilità 48% di riproporre un compito latente passato, con nuovi esempi.
- **16 neuroni nascosti; 48 update SGD × 2 esempi per compito; valutazioni separate da training**.
- **Quota totale identica** nei sei bracci appaiati: 12 unità rinnovate per 9 passaggi, massimo 3 per passaggio.
- I controller apprendono esternamente 5 pesi numerici; possono richiedere 0 / metà / massimo. I dettagli del compito attuale (incluso status novel/return) non vengono rivelati alla policy.
- 9 bracci in totale, comprendendo ablation dello stato, quota uniforme, permutata, sostituzione casuale delle unità, nessun rinnovo, massimo fisso, nuovo modello.

*AUC qui significa area normalizzata sotto la curva di accuratezza durante il task, non area ROC.*

## Risultati complessivi

| Strategia | AUC regolare | AUC distribuzione parzialmente nuova | Budget unità rinnovate per sequenza |
| --- | ---: | ---: | ---: |
| **Adaptive (stato + tempo)** | **73,89%** | **65,90%** | **12** |
| **Time-only (solo tempo)** | **74,66%** | **66,29%** | **12** |
| Stessa policy senza informazioni di stato | 74,62% | 65,85% | 12 |
| Quota uniformemente distribuita | 74,28% | 65,88% | 12 |
| Quota temporale permutata | 74,12% | 65,89% | 12 |
| Stessa quota/tempo, unità scelte casualmente | 74,79% | 64,98% | 12 |
| Nessun rinnovo | 73,20% | 65,84% | 0 (non dose matched) |
| Rinnovo massimo fisso | 75,11% | 66,05% | 27 (non dose matched) |
| Nuova rete per ogni compito | 73,67% | 62,98% | reset totale (non dose matched) |

**Metrica primaria**, test parzialmente nuovo, adaptive meno time-only:
- Differenza media appaiata **−0,39 punti percentuali**.
- Intervallo bootstrap percentile descrittivo 95% **da −1,23 a +0,33 punti**.
- **5 seed** migliori per adaptive, **7** peggiori. Intervallo comprende zero: nessun vantaggio robusto.

**Verifica secondaria** sul test regolare:
- Adaptive meno time-only = **−0,77 punti percentuali**.
- Intervallo bootstrap 95% **da −1,36 a −0,26 punti**.
- **2 seed** migliori per adaptive, **10** peggiori. In queste famiglie la policy state-aware risulta peggiore, non migliore.

**Altri confronti**: su shifted, adaptive meno quota permutata ≈ **+0,01 pp**, intervallo **−0,76 / +0,80 pp**; adaptive meno quota uniforme ≈ **+0,02 pp**, intervallo **−0,67 / +0,83 pp**. Non c'è evidenza che il tempismo appreso sia preferibile a semplici schedule.

L'accuratezza finale sullo shifted è **71,01% adaptive** contro **71,53% time-only**. La dimenticanza media è **12,45 pp adaptive** contro **13,00 pp time-only**: più memoria conservata non corrisponde qui a migliore AUC.

## Il comportamento cambia davvero con lo stato? Sì. Serve? Non dimostrato.

Il controller adattivo selezionato ha coefficienti approssimativi:

- bias 1,0051; errore attuale 1,0577; errore passato 1,4245;
- inattività −0,3526; indice di compito −1,5045.

Il controller di confronto time-only ha pesi [3, 0, 0, 0, 0]: tende a concentrare i 12 rinnovi nelle prime quattro transizioni.

**Le informazioni interne hanno un effetto causale sulle richieste della policy**:
- Test regolare: **75 richieste su 108** cambierebbero nascondendo gli input di stato.
- Test parzialmente nuovo: **59 richieste su 108** cambierebbero.
- Il sistema ha dovuto **forzare 21 rinnovi** nel test regolare e **23** nello shifted rispetto alle richieste, per soddisfare il vincolo di quota. Tali cambiamenti *non contano* come volontà o adattamento appreso.

Le 12 sequenze adattive per ciascun test hanno programmi di rinnovo effettivi differenti. La stessa policy privata dello stato segue invece uno schedule sostanzialmente fisso (per esempio 3,3,3,2,1,0,0,0,0). Ciò prova la dipendenza operativa delle azioni dagli input, **non** la convenienza scientifica di queste azioni.

L'ablation (stessa policy, caratteristiche di stato azzerate) ottiene **65,85%** sullo shifted contro **65,90%** del controller completo. Il suo effetto sulla performance è quindi trascurabile entro la variabilità osservata.

## Novità e ricorrenza

Sul totale delle 12 sequenze finali per split, per il solo controller adattivo, sono stati osservati:
- **73 task nuovi** (inclusi 12 task iniziali), **47 ritorni** di un task latente precedente in ciascuno split;
- dopo il task iniziale: **61 nuove occorrenze** e **47 ritorni**, per 108 passaggi valutati.

AUC media stratificata del controller:
- **Test regolare:** compiti nuovi **69,33%**, compiti ricorrenti **79,79%**.
- **Test shifted:** compiti nuovi **61,83%**, compiti ricorrenti **71,19%**.

La differenza è descrittiva: **tipi e difficoltà dei task potrebbero essere sbilanciati**, e la ricorrenza non è un intervento randomizzato a parità di identico task in tutte le condizioni. Non basta a dimostrare riconoscimento consapevole della novità, curiosità o recupero ottimale della memoria.

## Limiti e audit

1. **24 seed** di test, separati fra i due ambienti; ciascuno split ha solo **12 repliche**, intervalli ancora ampi.
2. Il mondo cambia ordine, novità e ricorrenza; resta tuttavia un compito supervisionato semplice in uno spazio 2D. La generalizzazione parziale checker/wedge non dimostra intelligenza aperta.
3. Il controllore viene ottimizzato da un **evolutore esterno** e non modifica codice, regole di aggiornamento SGD o obiettivi.
4. L'azione è limitata a rigenerare 0 / metà / massimo di unità ReLU: poca espressività per scoprire strategie sofisticate.
5. Il budget numerico degli interventi è identico, ma **non è stato misurato il costo CPU reale**, e il controller completo richiede osservazioni aggiuntive.
6. La metrica del riepilogo originale chiamata novel_events/return_events conta **soltanto il primo seed di ciascuno split** (errore diagnostico, non della fitness o delle AUC). I conteggi corretti sono ricavati dal file originale task_metrics.csv e indicati sopra: **73 / 47 eventi per split**. **La conclusione primaria non dipende da tale campo diagnostico.**
7. Le misure di incertezza sono descrittive bootstrap sulle differenze appaiate per seed; non c'è pretesa di significatività globale né correzione dei confronti multipli.

## Decisione scientifica

**Ipotesi M3 non confermata**: in un mondo che introduce novità e ricorrenze stocastiche, una policy che osserva la rete modifica effettivamente le proprie decisioni, ma non migliora in modo dimostrabile l'apprendimento rispetto a una policy che conosce soltanto il tempo, neppure su due famiglie di classificazione parzialmente nuove.

**Non cambiare retroattivamente le impostazioni** dopo aver visto questi risultati. Le difficoltà potrebbero riguardare lo spazio delle azioni, la capacità di usare memoria, i segnali osservabili oppure il semplice fatto che in questo mondo una schedulazione basata sullo stato non offre abbastanza vantaggio.

La prossima ipotesi dovrebbe essere separata e preregistrata: rendere **l'utilità del rinnovo causalmente verificabile** con perturbazioni calibrate, test di trasferimento più ampi, memoria esplicita e misure di overhead. Solo dopo risultati ripetibili ha senso parlare di evoluzione delle regole di apprendimento, non soltanto del momento del rinnovo.

## Dati originari

[GitHub Actions #44884705352](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44884705352), artefatto **m3-novelty-recurrence**, ID **14699671312**, scadenza prevista **21 dicembre 2026**.

- summary.json: policy scelte, AUC, intervalli e contatori.
- meta_history.jsonl: evoluzione e selezione dei due controller.
- test_seed_metrics.csv: **216 righe** (24 sequenze × 9 bracci).
- task_metrics.csv: **2.160 righe** (24 × 10 task × 9 bracci), con tag novel/return.
- [CSV compatto dei 24 seed](M3_test_seed_auc_2026-11-21.csv): tutte le nove AUC per sequenza, conservate nel repository.

Il file CSV e questo rapporto restano su GitHub anche dopo la scadenza degli artefatti Actions.
