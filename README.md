# AI Tutor Project

**1. Project Overview**
The Curriculum-Based AI Tutor is a retrieval-augmented question-answering system designed to answer student questions
using NCERT Class 8 Science textbook content. The objective is to provide concise, grade-appropriate answers while
restricting the knowledge source to the prepared textbook corpus. The system combines semantic search, FAISS retrieval, a
language model, and a Streamlit chat interface.
The project follows a Retrieval-Augmented Generation (RAG) workflow. Relevant textbook chunks are retrieved first and
supplied to the language model as context. This makes the answer generation more traceable and supports the display of
textbook source snippets to the student.

**2. Data Preparation and Corpus**
The downloaded NCERT Class 8 Science PDF was processed in the Jupyter Notebook. The workflow included PDF text
extraction, cleaning, chapter/page organization, and text chunk creation. The cleaned data was saved as
class8_science.jsonl for indexing and later application use.
Final corpus validation produced 307 text chunks and 18 unique chapter entries in the prepared corpus. Chunk sizes
ranged from 268 to 2,267 characters, with an average of approximately 1,393 characters.
Scope note: The project brief specifies 13 chapters, while the actual final validation shown during implementation contained
18 chapter entries. The implementation result is reported as observed; the final corpus should be checked against the exact
textbook edition before submission.

**3. Embeddings and Vector Index**
Each corpus chunk was converted into a semantic vector using sentence-transformers/all-MiniLM-L6-v2. The
embeddings have 384 dimensions. FAISS was used for nearest-neighbour search, with the indexed vectors associated with
the prepared textbook documents.
<img width="598" height="190" alt="image" src="https://github.com/user-attachments/assets/d40b8718-1610-4af8-b65d-df0c466b3a2f" />


**4. RAG Pipeline and AI Tutor**
For each student question, the application generates a query embedding and searches the FAISS index for the top three
relevant textbook chunks. A similarity threshold helps reject queries that are not sufficiently related to the prepared textbook
content. Retrieved context is then inserted into a textbook-only prompt.
The answer-generation stage uses google/flan-t5-small. The prompt instructs the model to use only the supplied textbook
context, avoid invented facts, use simple Class 8 language, and state when sufficient textbook information cannot be found.
The Streamlit interface maintains conversation history through session state and displays chapter, page, similarity score,
and source snippets.

**5. Evaluation**
The implemented evaluation used 10 test questions. Generated answers were compared with reference answers using
BLEU and ROUGE-L. A brief human review recorded correctness and relevance. The observed results were:
<img width="594" height="198" alt="image" src="https://github.com/user-attachments/assets/e1373701-d6af-4f41-80fa-9e4067b09e17" />


**6. Results and Error Analysis**
The human-review results indicate that the tested answers were generally relevant and reasonably correct when suitable
textbook context was retrieved. The average relevance score was 5.0/5 and the average correctness score was 4.0/5.
The relatively low BLEU (0.0116) and ROUGE-L (0.0962) scores indicate limited lexical overlap between generated and
reference answers. A semantically correct answer can use different wording, so these metrics should be interpreted together
with human review and retrieval similarity. The results also suggest that the evaluation set, prompting, retrieval strategy, and
generation model can be improved.
An error-analysis stage was completed and interactions were logged for later inspection. The logs can help identify weak
retrieval, low-overlap answers, and questions requiring better prompt design or corpus coverage.

**7. Streamlit Deployment**
A single-page Streamlit application was developed as the demonstration interface. It provides a dark-themed chat
dashboard, session-based conversation history, semantic retrieval, generated answers, textbook source snippets, similarity
information, and an out-of-syllabus fallback. Interactions are recorded for later error analysis.

**8. Conclusion**
The project demonstrates a complete curriculum-focused RAG pipeline: textbook extraction and cleaning, semantic
embeddings, FAISS indexing, contextual retrieval, constrained answer generation, evaluation, logging, and Streamlit
deployment. The approach provides a practical educational assistant whose answers can be traced to retrieved textbook
content.

**9. Future Work**
• Verify and align the final corpus with the exact 13-chapter scope specified in the project brief.
• Expand evaluation from 10 to 20 diverse questions, covering definitions, explanations, comparisons, and application-style
questions.
• Improve retrieval through better chunking, metadata filtering, and retrieval-quality evaluation.
• Experiment with a stronger instruction-tuned model while retaining textbook-only constraints.
• Apply the optional LoRA fine-tuning stage using curated question-answer pairs if additional alignment is needed.
• Strengthen out-of-syllabus detection and safeguards against unsupported answers.
Final deliverables: ai_tutor_class8.ipynb, class8_science.jsonl, evaluation.csv, app.py, interaction/error-analysis logs, and
this final report
