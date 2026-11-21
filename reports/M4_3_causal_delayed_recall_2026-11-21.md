# M4.3 — Memoria ritardata: prova causale positiva nel compito isolato, trasferimento ancora fragile

**Campagna:** 21 novembre 2026. **Esito primario preregistrato: POSITIVO.**  
**Intervento sullo stato ricorrente: POSITIVO.** **Controllo di inversione del segnale: POSITIVO.**  
**Generalizzazione a ritardi nuovi: NON CONFERMATA.** **Navigazione autonoma nell'ambiente: NON RISOLTA.**

Questo è il primo esperimento della linea M4 che supporta effettivamente l'uso causale di informazione passata nello **stato ricorrente della rete**, senza confondere il ricordo con la capacità di raggiungere fisicamente il bivio. Il risultato è reale **in questo benchmark semplice**; non è una prova di intelligenza ecologica generale, apprendimento sinaptico durante la vita o vita artificiale autonoma.

- [Protocollo M4.3 congelato prima del codice e della campagna](../docs/M4_3_PROTOCOL.md)
- [Codice sperimentale](../neuroevolution/m4_3.py), [test](../tests/test_m4_3.py), [PR #25 integrata in main](https://github.com/kkongiu/neuroevolution-lab/pull/25)
- [Esecuzione GitHub Actions #44909467363](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44909467363): validazione completa e campagna concluse con successo
- [Archivio permanente dei 16 gruppi seed genetico × split](M4_3_success_by_genetic_run_2026-11-21.csv)
- **Artefatto originale:** `m4-3-causal-delayed-recall` ID **14709625000**, scadenza prevista **21 dicembre 2026**. Contiene `summary.json`, `champions.json`, `train_validation.csv`, `recall_trial_metrics.csv`, `corridor_transfer.csv`, due schemi SVG e rispettivi JSON di osservazioni/decisioni.

## 1. Il problema causale lasciato aperto da M4.2

In M4.2 una rete ricorrente batteva la feedforward (**39,26%** contro **28,13%**) in un corridoio a T, ma raggiungeva anche il bivio più spesso (**72,07%** contro **56,25%**): un vantaggio di movimento poteva apparire come un vantaggio di memoria. Azzerare lo stato al bivio produceva una riduzione non conclusiva e il test con più ritardo collassava.

**M4.3 elimina il movimento dal test principale:** la rete riceve due osservazioni contenenti una direzione simbolica, poi **2, 4 o 6** osservazioni neutre identiche nei due mondi opposti; alla fine sceglie una delle **due direzioni legalmente disponibili** al bivio. Le azioni emesse prima della scelta sono ignorate; nessuna cambia osservazioni, energia, posizione o tempo. Tutti arrivano alla decisione. Per ogni seed vengono creati due obiettivi opposti su **identica geometria**.

La decisione finale vede la stessa osservazione nei due mondi: nessun odore, nessun cibo vicino, nessun segnale, nessun tempo esterno/energia leggibile. Non esiste pertanto una soluzione senza un canale di memoria interna, un meccanismo esterno o un cambiamento di stato dipendente dal vecchio cue.

La rete ricorrente Elman ha **8 neuroni tanh e 260 pesi**; la feedforward ha **11 neuroni e 268 pesi** (3,1% in più), ma nessun collegamento ricorrente. I pesi sono evoluti numericamente **fra vite**; le attivazioni ricorrenti cambiano **durante una vita**. Non vi è SGD né apprendimento dei pesi intra-vita.

## 2. Protocollo congelato e budget

- **Otto semi genetici indipendenti 90000–90007**, ogni seed con una ricerca per rete RNN e una per FF: **16 ricerche**.
- Per ciascuna ricerca: **8 generazioni ×14 genomi ×8 coppie training ×2 obiettivi ×3 ritardi = 5.376 prove training**, complessivamente **86.016 prove**.
- **Curriculum fissato prima dei test:** generazioni 0–1 ritardi (0,1,2), 2–3 (1,2,4), 4–7 (2,4,6). Ritardo 0 presenta il cue anche al momento della decisione: è un **supporto didattico del solo training**, assente nei test finali.
- Training: mondi 60000–60063, otto nuovi mondi per generazione; validazione: 61000–61007, sempre ritardi (2,4,6). I campioni RNN e FF di ogni seed vengono **scelti esclusivamente dalla fitness di validazione**.
- Fitness fissata: **0,85 × successo binario +0,15 × preferenza probabilistica corretta**. Le risposte reali sono visibili al sistema esterno di selezione genetica, **non** al modello durante la decisione. Nessun retuning sui seed finali.
- **Main indipendente:** 64 coppie di mondi 62000–62063, con ritardi 2/4/6 determinati da `seed % 3`. Due obiettivi opposti per coppia, sempre con stessa osservazione finale.
- **Shifted indipendente:** 32 coppie 63000–63031, ritardi nuovi 8/9/10, senza riaddestramento.
- **Trasferimento non primario:** 16 coppie spaziali 64000–64015, prima con percorso di movimento guidato, poi con movimento lasciato interamente al cervello.
- I **seed genetici** e i **world seed appaiati** sono le due dimensioni del bootstrap gerarchico descrittivo. Le due varianti di obiettivo non vengono trattate come campioni indipendenti.

## 3. Risultati principali: 64 coppie, otto evoluzioni indipendenti

| Agente o intervento | Successo nella scelta |
|---|---:|
| **RNN evoluta, stato ricorrente conservato** | **74,02%** |
| Feedforward evoluta, nessuna memoria | **50,00%** |
| **La stessa RNN, memoria azzerata appena prima della decisione** | **50,00%** |
| Stessa RNN, segnale iniziale invertito, obiettivo invariato | **25,98%** |
| Stessa RNN, cue eliminato | **50,00%** |
| Scelta casuale di uno dei due rami | **50,00%** |
| Memoria simbolica di un bit, programmata esplicitamente | **100,00%** |

| Confronto RNN normale meno… | Differenza media | Intervallo bootstrap gerarchico descrittivo 95% | Evoluzioni positive |
|---|---:|---:|---:|
| **Feedforward (PRIMARIO)** | **+24,02 pp** | **+16,50 / +31,54 pp** | **8 su 8** |
| **La stessa RNN con stato azzerato** | **+24,02 pp** | **+16,89 / +31,84 pp** | **8 su 8** |
| **La stessa RNN con cue invertito** | **+48,05 pp** | **+32,42 / +62,89 pp** | **8 su 8** |
| La stessa RNN senza cue | +24,02 pp | +16,50 / +31,15 pp | 8 su 8 |
| Scelta casuale binaria | +24,02 pp | +16,21 / +31,45 pp | 8 su 8 |
| Programma con memoria di un bit | **−25,98 pp** | **−33,11 / −18,95 pp** | 0 su 8 |

**Tutti e tre i gate preregistrati sono superati** sul main: vantaggio RNN versus FF, vantaggio RNN versus reset e danno da inversione del cue; la RNN supera il 55% assoluto e tutte le otto run genetiche sono positive.

Le otto RNN selezionate non performano tutte ugualmente: successo nel main **57,81%–86,72%**, media 74,02%. Il campione con risultato migliore **non** è stato scelto usando il test finale: ogni run contribuisce.

### Una precisazione metodologica importante: i tre controlli non sono tre repliche statisticamente indipendenti

Dato il disegno appaiato (stessa osservazione finale e due obiettivi opposti), **feedforward, rete con memoria resettata, assenza di cue e scelta casuale ottengono esattamente 50% per ogni coppia**, per costruzione matematica. Inoltre, invertire il cue in un problema binario simmetrico scambia l'esito: il successo medio con cue invertito è **1 − successo RNN**, e il suo contrasto è quindi collegato alla metrica primaria. **I tre gate sono controlli causali logicamente distinti ma non tre prove statistiche indipendenti.** Non gonfiamo l'evidenza sommando i tre intervalli come se provenissero da esperimenti separati.

Il controllo di azzeramento è comunque una **vera manipolazione fisica dello stato nascosto della stessa rete**: non cambia genoma, input di decisione, obiettivo o campione selezionato. L'esatto ritorno al 50% dopo l'intervento rende persuasiva l'interpretazione che lo stato interno trasportasse l'informazione del cue attraverso osservazioni neutre.

### I risultati per ritardo durante il main

| Ritardo neutro | Successo RNN | Feedforward/reset |
|---|---:|---:|
| 2 osservazioni | **72,92%** | 50,00% |
| 4 osservazioni | **78,87%** | 50,00% |
| 6 osservazioni | **70,45%** | 50,00% |

Le differenze fra 2, 4 e 6 **non** descrivono una curva pura di decadimento dello stesso mondo: ciascun ritardo è assegnato a world seed diversi. Il risultato positivo copre tutti i ritardi incontrati nel training, ma per una curva di durata rigorosa servirebbe assegnare **tutti i ritardi allo stesso insieme di mondi**.

## 4. Quando il ritardo è più lungo, il vantaggio non è confermato

Su **32 nuove coppie** con ritardi mai addestrati, 8/9/10:

| Agente | Successo |
|---|---:|
| RNN evoluta | **60,55%** |
| Feedforward, reset o nessun cue | **50,00%** |
| Stessa RNN con cue opposto | 39,45% |
| Memoria simbolica programmata | **100,00%** |

**RNN − FF:** +10,55 pp, intervallo gerarchico descrittivo **−0,39/+21,68 pp**, **6 run positive e 2 negative**.  
**RNN − reset:** +10,55 pp, intervallo **−0,59/+22,46 pp**, **non confermato**.  
**RNN − cue invertito:** +21,09 pp, intervallo **−0,39/+43,75 pp**, **non confermato**.

Tutti i gate scientifici della split shifted sono **falsi** secondo la regola fissata prima dell'esperimento. Le tre medie per ritardi 8/9/10 sono circa **58,52% / 57,39% / 66,25%**, ma il raggruppamento non usa lo stesso mondo per ogni ritardo: non è prova che dieci step siano più facili di otto. Due run genetiche scendono perfino **sotto il 50%** con ritardi lunghi, un segnale di fragilità fuori distribuzione.

**Conclusione corretta:** capacità di ricordare lo stimolo in intervalli incontrati durante la selezione, **non** robusta generalizzazione temporale oltre quei ritardi.

## 5. Transizione dalla memoria pura al corpo: cosa rimane?

Trasferimento a 16 nuovi labirinti a T per ciascuno degli otto genomi, sempre con due obiettivi opposti:

| Architettura | Percorso guidato, scelta finale | Movimento autonomo, successo complessivo | Arriva al bivio muovendosi autonomamente |
|---|---:|---:|---:|
| **RNN congelata** | **62,50%** | **23,83%** | 52,73% |
| Feedforward congelata | **50,00%** | **16,80%** | 50,78% |

Nel percorso guidato il motore esterno sposta l'organismo passo dopo passo, presentando sensori reali di pareti e cue: la rete deve ricordare il cue e decidere soltanto alla fine. Se invece controlla anche gli spostamenti, **il successo crolla al 23,83%**, sia per i fallimenti di arrivo sia per l'incapacità di scegliere correttamente dopo una storia sensoriale differente.

Il confronto fra 62,50% e 23,83% è **diagnostico, non un esperimento causale puro sul solo movimento**: lo scaffolding modifica anche la sequenza di osservazioni prima del bivio. Non si può attribuire tutto il calo a un unico fattore.

La rete feedforward rimane al 50% con guida, come previsto dall'identico input finale, e scende a 16,80% con movimento libero. La differenza fra i modelli in movimento libero non è il confronto primario né una prova di navigazione ecologica generalizzata.

## 6. Che cosa è dimostrato e che cosa non lo è

**Supportato dai dati:**
- Esistono otto RNN numericamente evolute che conservano informazione sul cue nel proprio stato ricorrente e, dopo una sequenza neutra, scelgono il lato corretto **oltre il caso**.
- Annullare la memoria interna nel medesimo genoma toglie questo vantaggio sul task.
- Il modello non può usare il movimento passato, l'energia, un orologio, un odore o la geometria finale per ricostruire quale dei due mondi opposti sia attivo.
- I risultati sono riproducibili da seed/genomi e sono positivi secondo i criteri fissati prima dei test principali.

**Non dimostrato:**
- Persistenza robusta a ritardi mai incontrati; i controlli shifted non superano l'intervallo.
- Trasferimento automatico della memoria al problema di muoversi e procurarsi cibo.
- Memoria di una mappa spaziale complessa: il compito usa un **simbolo cardinale**, assimilabile a pochi bit di memoria, non una rappresentazione topologica appresa.
- Apprendimento dei pesi **durante** la vita. Qui i pesi sono fissati durante il singolo episodio; la selezione numerica è esterna tra generazioni.
- Scelta autonoma degli obiettivi, riproduzione intra-ambiente, predatori, coscienza o vita emergente.

Gli intervalli sono **bootstrap descrittivi gerarchici**, campionando interi seed genetici e coppie mondo. Non vanno letti come garanzia di generalizzazione alla biologia o a ogni ambiente non osservato.

## 7. Conseguenza per la ricerca: M4.4

Il passo successivo non dovrebbe essere aggiungere nemici o riproduzione, ma **collegare una memoria già causalmente dimostrata a un problema motorio**. Una possibile M4.4 dovrebbe essere preregistrata con nuovi seed e includere:
1. Un **curriculum corporeo** a difficoltà crescente: percorso di avvicinamento completamente guidato, parzialmente guidato, poi pienamente libero, mantenendo lo stesso compito di scelta.
2. Confronti a budget identico fra architetture che hanno memoria e reti senza memoria, con reset al bivio, cue opposto e frazioni di successo di arrivo separate dalla correttezza della scelta.
3. Gli **stessi mondi valutati a più durate** del cue, per distinguere il vero decadimento temporale dalla variabilità della geometria.
4. Graduale sostituzione del simbolo di direzione perfetto con **landmark locali parziali**, e solo in una fase separata ricerca libera di cibo, energia rinnovabile e riproduzione.
5. Ogni claim scientifico deve ancora superare un controllo semplice (agente casuale legale, regola di memoria a un bit) e test OOD mai visti.

**Decisione finale:** M4.3 raggiunge l'obiettivo ristretto di provare **memoria ricorrente causalmente utile** nel compito controllato. Fallisce invece la generalizzazione forte a ritardi più lunghi e **non** dimostra una strategia ecologica autonoma. Conservare entrambi i risultati è la base per progettare il prossimo esperimento.

## Riproduzione e dati

L'artefatto originale [GitHub Actions #44909467363](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44909467363) comprende:
- `summary.json`: gate preregistrati, medie, ritardi e bootstrap gerarchici;
- `champions.json`: **16 genomi** (otto RNN e otto FF), valori numerici e generazioni scelte solo su validazione;
- `train_validation.csv`: **128 righe** (8 run ×2 architetture ×8 generazioni);
- `recall_trial_metrics.csv`: **10.752 prove di scelta** (8 run ×(64+32) mondi appaiati ×2 obiettivi ×7 bracci);
- `corridor_transfer.csv`: **512 prove trasferimento** (8 run ×16 coppie ×2 obiettivi ×2 architetture);
- `example_main.svg` e `example_shifted.svg`: geometria del percorso illustrativo **scaffold**, non traccia autonoma della rete; JSON di stimoli e scelte corrispondenti.

Il repository conserva permanentemente il [CSV compatto per run genetica](M4_3_success_by_genetic_run_2026-11-21.csv) e questo rapporto oltre alla possibilità di riprodurre l'esperimento con il comando `python -m neuroevolution.m4_3`.
