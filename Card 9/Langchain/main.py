from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os


def gerar_nomes_gatinhos(animal_type, color):

    llm = Ollama(model="mistral", temperature=0.6)

    prompt_gatinhos = PromptTemplate(
        input_variables=['animal_type', 'color'],
        template = "{animal_type} {color} acabou de ter 5 filhotes, me de sugestões de nomes para cada um deles."
    )

    gatinho_nomes_chain = LLMChain(
        llm = llm,
        prompt = prompt_gatinhos
    )

    response = gatinho_nomes_chain({'animal_type':animal_type, 'color':color})
    return response["text"]

if __name__=="__main__":
    print(gerar_nomes_gatinhos("gatinho", "azul"))