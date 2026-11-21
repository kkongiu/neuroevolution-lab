# M5.2A — Il collo di bottiglia è il movimento, non principalmente la riproduzione

**Campagna completata il 21 novembre 2026** (GitHub Actions, circa 15:38–15:39 UTC).

**Tracciabilità.** [Protocollo preregistrato prima del codice](../docs/M5_2A_PROTOCOL.md), commit originale 0fd024e98126ba7f22bf00e8ee2398876a05f2ae; [codice](../neuroevolution/m5_2a.py); [test](../tests/test_m5_2a.py); [workflow riuscito #44920201061](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44920201061); [CSV permanente con 30 mondi](M5_2A_world_survival_births_2026-11-21.csv); [PR #32](https://github.com/kkongiu/neuroevolution-lab/pull/32). **130 test superati**, 20 mondi main + 10 shifted, 400 tick e quattro bracci per mondo. Artefatto originale Actions: m5-2a-causal-factorial, ID 14713710782 (retention 30 giorni), con summary.json, world_metrics.csv, population_checkpoints.csv, SVG ed eventi genealogici.

## Esperimento

M5.1 dimostrava che gli animali programmati possono sopravvivere, mentre le reti con genomi casuali mutati ed ereditati muoiono. Ma cambiava simultaneamente il motore di ricerca del cibo e il criterio di riproduzione. M5.2A usa un disegno causale 2×2 sullo **stesso motore M5.0**, senza alterare energia, muri, rigenerazione, genealogia, costi, sensori o orizzonte. Una replica indipendente è il **mondo**.

| Braccio | Movimento | Riproduzione |
|---|---|---|
| **NN** | RNN originale | RNN originale |
| **NT** | RNN originale | Soglia programmata: energia almeno 46, età almeno 4 |
| **GN** | Greedy odore programmato, locale, senza memoria visite | RNN originale |
| **GT** | Greedy odore programmato | Soglia programmata |

La RNN a 401 pesi controlla esplicitamente ogni decisione quando prevista dalla configurazione. Dove la decisione è programmata, **non è un risultato di evoluzione neurale**. Tutti i mondi usano fondatori numerici inizialmente identici fra bracci, veri individui/figli, genomi ereditati, mutazione e medesimi tentativi di comparsa del cibo. La distribuzione realizzata del cibo diverge per effetto delle scelte degli animali.

**Seed congelati prima del test:** main 75000–75019; shifted 76000–76009, mai usati nelle precedenti campagne. Parametri principali M5.0: mondo 15×15, 20 fondatori, massimo 64, 20 cibi iniziali e 32 massimi; metabolismo 1/tick, cibo +18, età massima 100, costo di nascita per genitore 20 e 12 al figlio, 400 tick. Nel mondo shifted più muri, odore più debole e meno rigenerazione.

## Risultato primario preregistrato: presenza di almeno un vivente al tick 400

| Braccio | MAIN vivi su 20 | SHIFTED vivi su 10 | Popolazione finale media MAIN | Nascite medie MAIN | Cibi raccolti medi MAIN |
|---|---:|---:|---:|---:|---:|
| NN | **0** | **0** | 0,00 | 21,10 | 53,65 |
| NT | **1** | **0** | 0,05 | 22,80 | 67,55 |
| GN | **20** | **9** | 13,35 | 126,45 | 406,60 |
| GT | **20** | **10** | 12,95 | 111,70 | 406,60 |

**Risultato centrale:** correggere solo *la riproduzione* senza correggere movimento aiuta 1/20 mondi. Correggere solo *il movimento*, mantenendo la riproduzione neurale, fa sopravvivere **20/20 mondi**. Il forte collo di bottiglia nella configurazione originale è sensori-motorio/foraggiamento; non una scarsità globale di risorse e non principalmente il momento della riproduzione. Alla fine degli esperimenti main, NN e NT hanno entrambi una media di **32/32 cibi rimasti**. GN 3,85 e GT 3,25.

### Contrasti appaiati fra i 20 mondi MAIN

Endpoint binario sopravvivenza a 400 tick; intervalli **descrittivi** bootstrap percentile 95%, bootstrap sulla dimensione *mondo*, mai su singoli animali.

| Contrasto | Delta percentuale | Bootstrap 95% | Mondi positivi | Esito preregistrato |
|---|---:|---|---:|---|
| NT − NN: intervento sulla nascita con moto neurale | **+5 pp** | [0, +15] pp | 1/20 | **NON confermato** |
| GN − NN: intervento motorio con nascita neurale | **+100 pp** | [+100, +100] pp | 20/20 | Supportato |
| GT − NT: intervento motorio con nascita a soglia | **+95 pp** | [+85, +100] pp | 19/20 | Supportato |
| GT − GN: intervento sulla nascita con moto greedy | **0 pp** | [0, 0] pp | 0/20 | Nessun effetto su sopravvivenza main |
| GT − NN: entrambi gli interventi | **+100 pp** | [+100, +100] pp | 20/20 | Supportato |

Interazione su scala di sopravvivenza: **−5 pp**, intervallo [−15, 0] pp. Effetti soffitto: GN e GT sono entrambi 20/20 sopravvissuti, ma possono differire per durata individuale, rischi ecologici, produttività o natalità. Non è dimostrata equivalenza generale tra le due regole riproduttive.

### Stress test shifted (10 mondi nuovi)

Muri più densi, raggio di odore più breve e minore rigenerazione: NN **0/10**, NT **0/10**, GN **9/10**, GT **10/10**. Valori medi di cibo raccolto: rispettivamente 47,7; 41,8; 257,0; 292,9. Nascite medie: 19,6; 15,7; 71,4; 88,3. La riproduzione può ancora avere un impatto in alcuni mondi ostili: un mondo in cui GN muore è sostenuto da GT.

## Audit e limiti

- **Passa la verifica tecnica:** 130 test, identità esatta di NN rispetto a M5.0 e GT rispetto a M5.1 su seed comparabili, scelta dell'azione riproduttiva e motoria sottoposta a test causale, bilancio energetico chiuso e genealogie vere.
- **Passa l'intervento causale sul movimento:** GN–NN e GT–NT migliorano nettamente l'endpoint primario con confronti appaiati tra mondi.
- **Non passa l'intervento esclusivamente riproduttivo:** NT–NN migliora soltanto 1/20 mondi e il limite inferiore dell'intervallo include zero.
- **Non è dimostrata evoluzione motoria autonoma:** il motore che permette sopravvivenza è PROGRAMMATO. I genomi non apprendono tale abilità; la nascita e i cambiamenti genetici sono reali dentro la simulazione, ma non evidenziano adattamento motorio autonomo. Neppure GN prova la scelta riproduttiva neurale ottima: soltanto sufficiente quando il movimento è affidato allo script.

**Decisione scientifica:** passare a **M5.2B**, non scrivere un forager ancora migliore. Usare un controller neurale genuinamente evolvibile e più compatto, che veda gli stessi 15 sensori e produca indipendentemente quattro azioni motorie più RIPRODURSI. Preregistrare disegno, mutazioni sparse/totali/cloni/random, mondi e metriche su nuovi seed. Richiedere persistenza e **vantaggio genetico ereditario** contro nuovi genomi, più test causale del genotipo su mondi indipendenti. Nessuna alterazione post-hoc dei risultati di M5.0/M5.1/M5.2A.
