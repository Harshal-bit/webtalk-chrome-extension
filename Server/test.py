
import os
from dotenv import load_dotenv

from server import createDoc
import pandas as pd
from giskard.rag import KnowledgeBase
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from giskard.rag import generate_testset
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings




load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

documents = createDoc("https://www.ml.school/")

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="dbChroma"  # Specify the directory to persist the Chroma vector store
)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "dbChroma")


df = pd.DataFrame([d.page_content for d in documents], columns=["text"])
knowledge_base = KnowledgeBase(df)
print(df.head(10))
testset = generate_testset(
    knowledge_base,
    num_questions=60,
    agent_description="A chatbot answering questions about the Machine Learning School Website",
)

test_set_df = testset.to_pandas()

for index, row in enumerate(test_set_df.head(3).iterrows()):
    print(f"Question {index + 1}: {row[1]['question']}")
    print(f"Reference answer: {row[1]['reference_answer']}")
    print("Reference context:")
    print(row[1]['reference_context'])
    print("******************", end="\n\n")

testset.save("test-set.jsonl")


template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
print(prompt.format(context="Here is some context", question="Here is a question"))

retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("What is the Machine Learning School?")

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({"question": "What is the Machine Learning School?"})

def answer_fn(question, history=None):
    return chain.invoke({"question": question})

from giskard.rag import evaluate

report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)
display(report)
report.to_html("report.html")
report.correctness_by_question_type()
report.get_failures()
from giskard.rag import QATestset
testset = QATestset.load("test-set.jsonl")
test_suite = testset.to_test_suite("Machine Learning School Test Suite")
import giskard
def batch_prediction_fn(df: pd.DataFrame):
    return chain.batch([{"question": q} for q in df["question"].values])
giskard_model = giskard.Model(
    model=batch_prediction_fn,
    model_type="text_generation",
    name="Machine Learning School Question and Answer Model",
    description="This model answers questions about the Machine Learning School website.",
    feature_names=["question"], 
)
test_suite_results = test_suite.run(model=giskard_model)
display(test_suite_results)
import ipytest
import pytest
from giskard.rag import QATestset
from giskard.testing.tests.llm import test_llm_correctness


@pytest.fixture
def dataset():
    testset = QATestset.load("test-set.jsonl")
    return testset.to_dataset()


@pytest.fixture
def model():
    return giskard_model


def test_chain(dataset, model):
    test_llm_correctness(model=model, dataset=dataset, threshold=0.5).assert_()


