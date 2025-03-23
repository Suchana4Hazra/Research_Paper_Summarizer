# Research_Paper_Summarizer

# **Problem Statement 2: Research Article Summarization Using Advanced NLP Techniques**  
#### <div align="right">🎯 Marks: 60</div> 

#### 📌 Problem Statement:
With the exponential rise in scientific publications, researchers struggle to keep up with the vast volume of literature. The challenge lies in creating a **state-of-the-art model that accurately summarizes research articles** while preserving key insights and readability, enabling researchers to assimilate information swiftly. Scientific papers differ significantly from general text due to their **structured frmat (Introduction, Methods, Results, Discussion, etc.), citation dependencies, and inclusion of figure/tables**. Summarizing such documents requires handling **domain-specific challenges**, maintaining semantic coherence, and ensuring **faithful knowledge retention**. This competition challenges participants to develop a **framework for summarizing research artciels**, leveraging **state-of-the-art approaches**, preferably by using **Large Language Models (LLMs)** with an aim to develop a **hybrid summarization model** that surpasses existing benchmark models while maintaining computational efficiency. 

Participants must develop an **extractive-abstractive hybrid model** that:
- Summarized single and multi-document research papers
- Handles long-document summarization constraints while maintaining efficiency

Participants will be provided with **multiple research article datasets**, including a proprietary dataset from IIEST Shibpur, **CompScholar**, and two publicly available benchmark datasets (**PubMed and arXiv**). The task is to develop an efficient **text-based abstractive summarization model** that generates **concise, coherent, and relevant summaries** while outperforming existing models. Participants must also compare their models against **state-of-the-art summarization frameworks** (e.g., SummRuNNer, Pointer-Generator, Discourse Aware, BERT, T5, BART, PEGASUS, GPT-4, Longformer, and LED), providing a **detailed performance analysis with citations**.  

#### 📂 Dataset Link:
1. **CompScholar Dataset** ([Dataset Link](https://github.com/jayantapaul/BrainDead-2K25/blob/1dafe7a5b42a33e0afd5dfa183780ca32c036dad/Brain%20Dead%20CompScholar%20Dataset.csv)).  
2. **PubMed Dataset** ([Dataset Link 1](https://huggingface.co/datasets/ncbi/pubmed)) ([Dataset Link 2](https://huggingface.co/datasets/ccdv/pubmed-summarization))—contains millions of biomedical research articles with structured abstracts.  
3. **arXiv Dataset** ([Dataset Link](https://huggingface.co/datasets/arxiv-community/arxiv_dataset))—includes structured documents across multiple scientific disciplines.  

#### **Dataset Description**:
The datasets provided for this competition contain **full-text research articles** from diverse disciplines, including science, medicine, and engineering. Each dataset consists of **article abstracts and full texts**, with labeled summaries provided for supervised learning.  

The datasets contain the following:  
- **Research Articles**: Text files containing full-length papers and abstracts.  
- **Metadata**: Titles, keywords, citations, author information, and journal sources.  
- **Summarization Labels**: Ground truth summaries for evaluation and model training.

#### 📊 Metadata Description of 'CompScholar' file:

The **`CompScholar.csv`** file consists of the **metadata informations** instead of full-length article text for **370 research articles from a variety of domains**. 
-	**'Paper Id'**: Unique identifier for each research paper
-	**'Paper Title'**: The full title of the paper
-	**'Key Word'**: Keyword of the paper
-	**'Abstract'**: The full-length abstract from the research article
-	**'Conclusion'**: The conclusion section from the research article
-	**'Document'**: Combined previous metadata (Paper Title, Key Word, Abstract, Conclusion) and kept in the 'Document' attribute
-	**'Paper Type'**: This particular field states the type of paper with values like Text Summarization, Natural Language Processing, Text Generation, Medical Data, Neural Network, etc.
-	**'Summary'**: Manually annotated summary of that particular research article
-	**'Topic'**: The topic attribute specifies the major domain under which that partcular paper falls, like Natural Language Processing, Medical Data Analysis and Deep Learning and Machine Learning
-	**'OCR'**: Additional text information extracted from the diagrams, tables, images, output graphs using Optical Character Recognition (OCR) **[May not be needed in the summarization task]**
-	**'labels'**: Each of the paper has been assigned one label based on the content and domains in which it falls, **'Deep Learning and Machine', 'Medical Data Analysis', 'Object Recognition'** and **Sentiment Analysis**. **[May not be needed in the summarization task]**

#### 📊 Data Field Description of 'arXiv' file:

The **arXiv dataset** consists of **1.7 million arXiv articles** for applications like trend analysis, paper recommender engines, category prediction, co-citation networks, knowledge graph construction and semantic search interfaces.
```json
{
"id": "0704.0002",
"submitter": "Louis Theran",
"authors": "Ileana Streinu and Louis Theran",
"title": "Sparsity-certifying Graph Decompositions",
"comments": "To appear in Graphs and Combinatorics",
"journal-ref": "None",
"doi": "None",
"report-no": "None",
"categories": "math.CO cs.CG",
"license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
"abstract": "We describe a new algorithm, the $(k,\\ell)$-pebble game with colors, and use\nit obtain a characterization of the family of $(k,\\ell)$-sparse graphs and\nalgorithmic solutions to a family of problems concerning tree decompositions of\ngraphs. Special instances of sparse graphs appear in rigidity theory and have\nreceived increased attention in recent years. In particular, our colored\npebbles generalize and strengthen the previous results of Lee and Streinu and\ngive a new proof of the Tutte-Nash-Williams characterization of arboricity. We\nalso present a new decomposition that certifies sparsity based on the\n$(k,\\ell)$-pebble game with colors. Our work also exposes connections between\npebble game algorithms and previous sparse graph algorithms by Gabow, Gabow and\nWestermann and Hendrickson.\n",
"update_date": "2008-12-13"
}
```
- **'id'**: ArXiv ID (can be used to access the paper)
- **'submitter'**: Who submitted the paper
- **'authors'**: Authors of the paper
- **'title'**: Title of the paper
- **'comments'**: Additional info, such as number of pages and figures
- **'journal-ref'**: Information about the journal the paper was published in
- **'doi'**: Digital Object Identifier
- **'report-no'**: Report Number
- **'abstract'**: The abstract of the paper
- **'categories'**: Categories / tags in the ArXiv system

#### 📊 Data Field Description of 'PubMed' file (Dataset 1):

**PubMed** comprises more than **36 million citations** for biomedical literature from **MEDLINE**, life science journals, and online books. Citations may include links to full-text content from PubMed Central and publisher web sites.

```json
{
    "MedlineCitation": {
        "PMID": 0,
        "DateCompleted": {"Year": 0, "Month": 0, "Day": 0},
        "NumberOfReferences": 0,
        "DateRevised": {"Year": 0, "Month": 0, "Day": 0},
        "Article": {
            "Abstract": {"AbstractText": "Some abstract (can be missing)" },
            "ArticleTitle": "Article title",
            "AuthorList": {"Author": [
                {"FirstName": "John", "ForeName": "Doe", "Initials": "JD", "CollectiveName": ""}
                {"CollectiveName": "The Manhattan Project", "FirstName": "", "ForeName": "", "Initials": ""}
            ]},
            "Language": "en",
            "GrantList": {
                "Grant": [],
            },
            "PublicationTypeList": {"PublicationType": []},
        },
        "MedlineJournalInfo": {"Country": "France"},
        "ChemicalList": {"Chemical": [{
            "RegistryNumber": "XX",
            "NameOfSubstance": "Methanol"
        }]},
        "CitationSubset": "AIM",
        "MeshHeadingList": {
            "MeshHeading": [],
        },
    },
    "PubmedData": {
        "ArticleIdList": {"ArticleId": "10.1002/bjs.1800650203"},
        "PublicationStatus": "ppublish",
        "History": {"PubMedPubDate": [{"Year": 0, "Month": 0, "Day": 0}]},
        "ReferenceList": [{"Citation": "Somejournal", "CitationId": 01}],
    },
}
```

Main Fields will probably interest the participants are:
- "MedlineCitation" > "Article" > "AuthorList" > "Author"
- "MedlineCitation" > "Article" > "Abstract" > "AbstractText"
- "MedlineCitation" > "Article" > "Article Title"
- "MedlineCitation" > "ChemicalList" > "Chemical"
- "MedlineCitation" > "NumberOfReferences"

#### 📊 Data Field Description of 'PubMed' file ([Dataset 2](https://huggingface.co/datasets/ccdv/pubmed-summarization)):

- **'id'**: paper id
- **'article'**: a string containing the body of the paper
- **'abstract'**: a string containing the abstract of the paper

#### 📊 Data Splits of 'PubMed' file ([Dataset 2](https://huggingface.co/datasets/ccdv/pubmed-summarization)):

| Dataset Split | Number of Instances |
|------|-------|
| Train | 119,924 |
| Validation | 6,633 |
| Test | 6,658 |

The competition will focus on **extractive-abstractive summarization** techniques. Participants will explore **LLM-based** models to generate meaningful summaries that maintain contextual relevance.  

---

#### 📊 Deliverables:
- A **novel text summarization model** designed to generate concise, coherent, and accurate summaries of research articles.  
- **Trained models** demonstrating superior performance compared to existing benchmark summarization frameworks.  
- **Comparative results with state-of-the-art models**, including SummRuNNer, Pointer-Generator, Discourse Aware, BERT, T5, BART, PEGASUS, Longformer, LED, GPT-4-based summarization approaches, Other models cited in recent research papers  
- **A comprehensive evaluation report**, comparing results with existing benchmarks.  

#### **Performance Metrics**:
Models will be evaluated on:  
1. **ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)**—Measures the overlap of generated summaries with reference summaries.  
2. **BLEU Score**—Evaluates text fluency and translation quality.  
3. **Summarization Length and Readability**—Ensuring meaningful information is retained in concise form.  
4. **Computational Efficiency**—Assessing training time, inference speed, and memory usage.  

**Ranked Model Performance Table (for CompScholar Dataset)**:
| Rank | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |  
|------|-------|---------|---------|---------|------|  
| 1 | PEGASUS | 45.1 | 21.8 | 42.3 | 36.2 |  
| 2 | BART | 43.5 | 19.4 | 40.6 | 33.8 |  
| 3 | Longformer | 41.2 | 18.9 | 39.1 | 32.4 |  
| 4 | LED | 40.5 | 17.8 | 38.6 | 31.7 |  
| 5 | GPT-4-Summarization | 39.2 | 16.5 | 37.2 | 30.8 |  
