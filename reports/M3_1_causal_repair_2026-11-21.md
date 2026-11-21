# M3.1 — Valore causale del rinnovo neurale: primo benchmark positivo

**Data:** 21 novembre 2026  
**Esito operativo:** GitHub Actions completato con successo, test unitari e smoke test superati.  
**Esito scientifico:** **POSITIVO limitatamente a un benchmark causale artificiale, progettato come controllo positivo.** Non prova auto-modifica generale, plasticità continua o autonomia.

- [Protocollo preregistrato PRIMA della campagna](../docs/M3_1_PROTOCOL.md)
- [GitHub Actions — run #44897210983](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44897210983)
- [Pull request #9](https://github.com/kkongiu/neuroevolution-lab/pull/9), integrata in main
- [Codice M3.1](../neuroevolution/m3_1.py) e [test](../tests/test_m3_1.py)
- [CSV per ogni seed](M3_1_seed_metrics_2026-11-21.csv)
- Artefatto originale **m3-1-causal-repair** (ID **14704397992**), scadenza prevista **21 dicembre 2026**.

## 1. La domanda che abbiamo effettivamente posto

La M3 mostrava che una politica poteva reagire a osservazioni sullo stato di una rete senza ottenere un vantaggio verificabile su una semplice schedulazione temporale. In M3.1 abbiamo voluto stabilire un **controllo positivo causale**: quando conservare una rappresentazione acquisita e quando rinnovare unità neurali devono poter produrre effetti opposti, misurabili direttamente.

Per ogni modello inizialmente addestrato creiamo due situazioni **artificialmente costruite**:

1. **Intatta:** la rete conserva le sue connessioni dopo avere imparato il problema.
2. **Danneggiata:** invertiamo e amplifichiamo di cinque volte i pesi di uscita delle tre unità più influenti, introducendo un danno esogeno.

Per ciascuna di queste due situazioni, confrontiamo **entrambi gli esiti potenziali**: mantenere le connessioni o rinnovare tre unità selezionate con una regola fissa. Le due alternative partono dallo **stesso identico stato prima dell'intervento**, ricevono gli stessi esempi e lo stesso budget di aggiornamenti SGD. La differenza di accuratezza durante l'apprendimento è pertanto direttamente attribuibile all'intervento, entro questo simulatore.

**Precisazione fondamentale:** il danno colpisce le unità più influenti e l'operatore di rinnovo individua proprio le unità più influenti con una procedura deterministica. È una corrispondenza intenzionale per costruire un benchmark che distingua le azioni. Il sistema non deve scoprire da zero come rigenerare neuroni.

## 2. Configurazione e isolamento dei dati

- Rete MLP a **12 unità ReLU**; intervento massimo di **3 unità**.
- Pre-addestramento: **96 aggiornamenti SGD**, batch da 4; dopo l'intervento altri **48 aggiornamenti** identici per ogni alternativa.
- Valutazione su **96 esempi holdout** ai passi 0/12/24/36/48. **AUC** = area normalizzata sotto la curva di accuratezza *durante l'apprendimento*, **non** ROC AUC.
- La regola appresa vede **solo l'accuratezza su 48 esempi etichettati del training** prima dell'intervento. Non conosce il flag «intatta/danneggiata» né l'effetto controfattuale sui dati finali.
- La regola scelta è semplicemente: **rinnova se l'accuratezza del probe di training è inferiore all'80%**.
- Il controllo time-only viene ottimizzato indipendentemente su training e validazione, ma può osservare soltanto l'indice dell'episodio, non la precisione del modello.
- Meta-training **8 seed: 12000–12007**; meta-validation **4 seed: 13000–13003**.
- Final test **24 seed: 14000–14023**, ciascuno con 4 modelli iniziali × 2 contesti = **192 decisioni appaiate**.
- Test OOD parziale **16 seed: 15000–15015**, con XOR/cerchio al posto dei classificatori lineari = **128 decisioni**.
- I risultati finali NON sono usati per modificare la soglia, l'operatore, la gravità del danno, le famiglie di problemi o il criterio statistico.

## 3. Verifica causale su meta-training, effettuata prima del test finale

Il protocollo richiedeva una condizione preliminare: se il rinnovo non penalizzava mediamente le reti intatte e non aiutava mediamente quelle danneggiate, il benchmark non avrebbe dato una prova utile.

| Effetto causale medio AUC (rinnovo − conservazione) | Meta-training |
|---|---:|
| Reti intatte | **−5,90 punti percentuali** |
| Reti danneggiate | **+34,18 punti percentuali** |

**Calibrazione superata.** È una proprietà del nostro mondo costruito, non una scoperta sulla plasticità delle reti in generale.

## 4. Test finale indipendente principale

| Strategia | AUC media sui 24 seed | Interventi per sequenza |
|---|---:|---:|
| **Regola appresa dall'accuratezza del training** | **90,51%** | **4 su 8** |
| **Controllo rimescolato a dose identica** | **79,69%** | **4 su 8** |
| Controllore time-only (ottimizzato separatamente) | 85,39% | 8 su 8 |
| Non rinnovare mai | 75,31% | 0 su 8 |
| Rinnovare sempre | 85,39% | 8 su 8 |
| Oracle che conosce ex post entrambi i risultati | 90,55% | variabile |

### Confronto primario preregistrato: appreso vs rinnovi rimescolati

- **Differenza media: +10,82 punti percentuali di AUC.**
- Intervallo bootstrap descrittivo 95%: **+9,28 / +12,33 punti**.
- **24 seed positivi su 24**, nessuno negativo o in pareggio.
- Il controllo rimescolato sostituisce **esattamente lo stesso numero** di unità per seed: quindi il vantaggio non si spiega con il solo numero complessivo di rinnovi.

### Controllo necessario: appreso vs time-only

- **+5,12 punti percentuali di AUC**.
- Intervallo bootstrap descrittivo 95% **+4,29 / +5,98 punti**.
- Anche qui **24 seed positivi su 24**.

Il tempo, nel nostro benchmark, non predice il danno. Il controllo temporale selezionato è pertanto degenerato nella regola **«rinnova sempre»**. Questa alternativa è forte nel riparare il danno ma danneggia la rappresentazione quando la rete è già intatta.

L'oracle raggiunge il **90,55%** contro **90,51%** della regola appresa. La regola sceglie l'azione dal risultato controfattuale migliore in **186 decisioni su 192 (96,88%)**.

### Stima dell'effetto causale dell'azione

| Stato prima dell'intervento | Effetto medio AUC (rinnovo − conservazione) |
|---|---:|
| Rete intatta | **−10,24 pp** |
| Rete danneggiata | **+30,39 pp** |

Il risultato prova che **la scelta dell'intervento conta in modo causalmente osservabile dentro questo benchmark**. Non è semplicemente una curva che cresce dopo più generazioni evolutive.

## 5. Verifica separata su famiglie di problemi diverse

Senza modificare la regola dell'80%, abbiamo ripetuto l'esperimento su **16 nuovi seed** che includono problemi XOR e a cerchio.

| Strategia | AUC media OOD parziale |
|---|---:|
| **Regola appresa** | **73,00%** |
| Time-only / rinnovo sempre | 71,27% |
| Rimescolamento a dose identica | 70,20% |
| Conservazione sempre | 63,54% |
| Oracle | 74,57% |

- Appresa − rimescolata: **+2,80 pp**, bootstrap descrittivo 95% **+1,13 / +4,60 pp**; **9 seed positivi, 6 pari, 1 negativo**.
- Appresa − time-only: **+1,73 pp**, bootstrap descrittivo 95% **+0,75 / +2,88 pp**; **7 positivi, 9 pari, 0 negativi**.
- La correttezza delle scelte rispetto all'oracle scende drasticamente dal **96,88%** al **65,63%**.
- In questa distribuzione il controller interviene **6,94 volte su 8** per sequenza, anziché 4. In particolare rinnova **47/64** reti intatte e tutte e 64 le reti danneggiate.

**Il trasferimento è quindi parziale e fragile.** La soglia 80%, scelta sui classificatori lineari, tende a interpretare come «guasto» anche una rete intatta che sta soltanto affrontando una famiglia di problemi più difficile.

## 6. Un punto critico: il controller non possiede ancora una vera autorappresentazione

La distinzione dei contesti nel test principale era relativamente facile: **accuratezza di training prima dell'intervento 95,01% per reti intatte, 16,23% per reti danneggiate**. Un solo indicatore superficiale di errore è stato sufficiente a riconoscere praticamente sempre il guasto.

Su XOR/cerchio l'accuratezza preliminare delle reti intatte scende al **76,33%** e quella delle reti danneggiate è **32,75%**: la soglia separa molto peggio i contesti. Questo spiega plausibilmente il calo del riconoscimento, senza dimostrarne la causa in modo esaustivo.

Il controller:
- NON ispeziona direttamente una rappresentazione delle proprie necessità di apprendimento: osserva soprattutto un punteggio su dati etichettati;
- NON inventa l'operatore che sceglie quali neuroni rinnovare;
- NON genera obiettivi o decide autonomamente cosa considerare successo;
- NON modifica il codice né la regola SGD; il suo comportamento è una soglia numerica selezionata esternamente.

Il benchmark è dunque paragonabile al **riconoscimento di un guasto e all'attivazione di un intervento di manutenzione**, non a una mente che si riprogetta autonomamente.

## 7. Limiti, riproducibilità e decisione

- I 192 casi del test principale sono correlati: **l'unità statistica indipendente è il seed**, non ciascuna alternativa sperimentale.
- Il danno è **artificiale** e deliberatamente coordinato con una procedura di rinnovo delle unità influenti. È un benchmark di *positive control*, non un ambiente naturale.
- Le reti sono addestrate indipendentemente in ogni trial: non abbiamo studiato una singola rete che mantiene identità e memoria lungo una vera vita continua.
- Abbiamo confrontato lo stesso numero di unità rinnovate con lo shuffle, ma non il **tempo CPU totale** per la diagnostica della regola.
- La replicazione con famiglie diverse non rappresenta una completa generalizzazione fuori distribuzione: anche lì l'intervento artificiale e la regola che individua le unità restano gli stessi.
- L'«oracle» usa risultati holdout **dopo** averli osservati: è soltanto un limite superiore diagnostico, non una strategia utilizzabile.

**Conclusione:** M3.1 realizza per la prima volta una verifica positiva e riproducibile del *valore della scelta* di mantenere o rinnovare neuroni, ma solo in un ambiente in cui questa scelta è stata resa informativamente accessibile e utile per costruzione. Questo non smentisce gli esiti negativi di M2B e M3, che rimangono pubblicati.

### Prossimo gate suggerito — M3.2

Prima di introdurre reti più grandi, verificare se un modello può diagnosticare **perdita di plasticità non etichettata** senza danni artificiali e senza conoscere in anticipo l'operatore ottimale. Separare un problema semplicemente difficile da una struttura realmente compromessa, usare compiti ricorrenti e memoria verificabile, e confrontare il valore di interventi sotto uguale limite computazionale. Preregistrare nuovi seed e una politica capace di utilizzare segnali delle rappresentazioni interne, non soltanto l'accuratezza osservata.

**Non aggiornare retroattivamente i risultati M3.1 né riutilizzare i seed di test 14000–15015 per scegliere la M3.2.**

## Provenienza

L'artefatto originale [GitHub Actions #44897210983](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44897210983) contiene:
- `summary.json`: parametri, criteri fissati, effetti causali, bootstrap per seed e condizioni;
- `seed_metrics.csv`: **40** righe (24 main + 16 shifted), prestazioni di ogni strategia;
- `counterfactuals.csv`: **320** confronti appaiati (192 main + 128 shifted), ognuno con entrambe le AUC potenziali, decisione del controllore, controllo temporale e dose-matched.

Le statistiche principali e i risultati per ciascun seed sono conservati permanentemente in questo rapporto e nel [CSV archivio](M3_1_seed_metrics_2026-11-21.csv), anche dopo la scadenza dell'artefatto ZIP su GitHub.
