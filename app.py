from flask import Flask, request, render_template, jsonify
import requests
import openai
import xml.etree.ElementTree as ET
import webbrowser
from threading import Timer
import re
from sentence_transformers import SentenceTransformer
import hdbscan
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)

# Configura la chiave API di OpenAI
openai.api_key = "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"  # Sostituisci con la tua chiave API reale

# Imposta una chiave segreta per le sessioni
app.secret_key = 'd394bef95bc4113c308e79f29e8bcb9b2ae0036be9ae2473'  # Sostituisci con una chiave generata in modo sicuro

# Lista di journal accettati
accepted_journals = [
    "Nature", "Science", "Cell", "The Lancet", "Nature Reviews Drug Discovery",
    # Aggiungi qui gli altri journal come nella tua lista
]

# Variabile globale per memorizzare le idee innovative
innovative_ideas = []

# Funzione per preprocessare i testi (pulizia)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Rimuove caratteri speciali
    return text

# Inizializza sessione con retry per gestire errori di connessione
def requests_retry_session(retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Funzione per esplorare ArXiv API (con parsing XML)
def explore_arxiv(need):
    url = f"http://export.arxiv.org/api/query?search_query=all:{need}&start=0&max_results=20"
    response = requests_retry_session().get(url, timeout=10)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        relevant_papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            relevant_papers.append({
                'title': title,
                'abstract': summary,
                'journal': 'arXiv',
                'content': f"{title} {summary}"
            })
        print(f"Number of papers from ArXiv: {len(relevant_papers)}")
        return relevant_papers
    else:
        print(f"Error fetching from ArXiv API: {response.status_code}")
        return []

# Funzione per esplorare PubMed API
def explore_pubmed(need):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={need}&retmax=20&retmode=json"
    try:
        response = requests_retry_session().get(url, timeout=10)
        if response.status_code == 200:
            ids = response.json().get('esearchresult', {}).get('idlist', [])
            relevant_papers = []
            for pmid in ids:
                details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                try:
                    details_response = requests_retry_session().get(details_url, timeout=10)
                    paper_info = details_response.json().get('result', {}).get(pmid, {})
                    title = paper_info.get('title', 'No title')
                    journal = paper_info.get('source', 'Unknown Journal')
                    abstract = paper_info.get('abstract', 'No abstract available')
                    relevant_papers.append({
                        'title': title,
                        'abstract': abstract,
                        'journal': journal,
                        'content': f"{title} {abstract}"
                    })
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching paper details for PMID {pmid}: {e}")
            print(f"Number of papers from PubMed: {len(relevant_papers)}")
            return relevant_papers
        else:
            print(f"Error fetching from PubMed API: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from PubMed: {e}")
        return []

# Funzione per combinare tutte le fonti
def explore_papers(need):
    arxiv_papers = explore_arxiv(need)
    pubmed_papers = explore_pubmed(need)
    
    all_papers = arxiv_papers + pubmed_papers
    print(f"Total papers found: {len(all_papers)}")
    for paper in all_papers:
        print(f"Paper: {paper['title']}, Journal: {paper['journal']}")
    
    return all_papers



# Inizializza il modello di embedding BERT pre-addestrato
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Funzione per il clustering avanzato con HDBSCAN
def advanced_clustering(papers):
    if not papers:
        print("No papers to cluster")
        return {}

    documents = [preprocess_text(paper['content']) for paper in papers]
    embeddings = bert_model.encode(documents)
    print(f"Embeddings shape: {embeddings.shape}")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    print(f"Cluster labels: {cluster_labels}")
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        cluster_name = "Noise" if label == -1 else f"Cluster {label + 1}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(papers[i])
    
    print(f"Clusters formed: {len(clusters)}")
    for cluster_name, cluster_papers in clusters.items():
        print(f"{cluster_name} contains {len(cluster_papers)} papers")
    
    return clusters


# Funzione per generare idee da cluster di paper
def generate_ideas_from_clusters(clusters):
    ideas = []
    for cluster_name, cluster_papers in clusters.items():
        if cluster_papers and cluster_name != "Noise":
            combined_titles = ', '.join([paper['title'] for paper in cluster_papers])
            abstracts = [paper['abstract'] if paper['abstract'] != 'No abstract available' else f"No abstract available but the title is '{paper['title']}'" for paper in cluster_papers]
            combined_abstracts = ' '.join(abstracts)

            prompt = f"Given the following papers: {combined_titles}, with abstracts: {combined_abstracts}, generate an innovative business idea that integrates insights from these papers related to '{cluster_name}'."
            
            print(f"Prompt sent to OpenAI for cluster '{cluster_name}': {prompt}")

            retries = 5
            for i in range(retries):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates business ideas from clusters of research papers."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200
                    )
                    idea = response.choices[0].message['content'].strip()
                    print(f"Idea generated for cluster '{cluster_name}': {idea}")
                    ideas.append({
                        'cluster': cluster_name,
                        'idea': idea,
                        'papers': cluster_papers
                    })
                    break
                except Exception as e:
                    print(f"Error during idea generation: {e}")
                    break
    return ideas


# Route per il form dell'input e output
@app.route('/', methods=['GET', 'POST'])
def index():
    global innovative_ideas  # Rendi la variabile globale accessibile
    if request.method == 'POST':
        need = request.form.get('need')
        print(f"Received need: {need}")

        all_papers = explore_papers(need)
        print(f"Total papers found: {len(all_papers)}")  # Debug statement

        if not all_papers:
            return jsonify({
                'papers_used': [],
                'innovative_ideas': [],
                'message': "Nessun paper trovato. Prova con un altro bisogno aziendale."
            })

        clusters = advanced_clustering(all_papers)
        print(f"Clusters formed: {clusters}")  # Debug statement

        innovative_ideas = generate_ideas_from_clusters(clusters)
        print(f"Total innovative ideas generated: {len(innovative_ideas)}")  # Debug statement

        return render_template('ideas.html', ideas=innovative_ideas)  # Renderizza la pagina delle idee

    return '''
        <form method="POST">
            Inserisci il bisogno aziendale: <input type="text" name="need">
            <input type="submit" value="Cerca paper e genera idee">
        </form>
    '''


@app.route('/ideas')
def list_ideas():
    global innovative_ideas  # Rendi la variabile globale accessibile
    return render_template('ideas.html', ideas=innovative_ideas)  # Renderizza la pagina con le idee

@app.route('/idea/<int:idea_id>')
def idea_detail(idea_id):
    global innovative_ideas  # Rendi la variabile globale accessibile
    if idea_id < 0 or idea_id >= len(innovative_ideas):
        return "Idea not found", 404  # Gestisci l'errore se l'ID non è valido
    idea = innovative_ideas[idea_id]  # Recupera l'idea in base all'ID
    return render_template('idea_detail.html', idea=idea)  # Renderizza la pagina con i dettagli

# Funzione per aprire automaticamente il browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()  # Apre il browser dopo 1 secondo
    app.run(debug=True)
