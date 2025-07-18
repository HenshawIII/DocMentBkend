import { ChatOpenAI } from "@langchain/openai"
import { PromptTemplate } from "@langchain/core/prompts"
import dotenv from "dotenv"
import { vectorStore,model,embeddings } from "./index.js"
import { StringOutputParser } from "@langchain/core/output_parsers"
import { RunnableSequence ,RunnablePassthrough} from "@langchain/core/runnables"

dotenv.config()

function combineDocs(docs) {
    return docs.map(doc => doc.pageContent).join("\n\n")
} 

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: `

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const punctuationTemplate = `Given a sentence, add punctuation where needed. 
    sentence: {sentence}
    sentence with punctuation:  
    `
const punctuationPrompt = PromptTemplate.fromTemplate(punctuationTemplate)

const grammarTemplate = `Given a sentence correct the grammar.
    sentence: {punctuated_sentence}
    sentence with correct grammar: 
    `
const grammarPrompt = PromptTemplate.fromTemplate(grammarTemplate)

const translationTemplate = `Given a sentence, translate that sentence into {language}
    sentence: {grammatically_correct_sentence}
    translated sentence:
    `
const translationPrompt = PromptTemplate.fromTemplate(translationTemplate)

const retriever = vectorStore.asRetriever()
const tweetTemplate = 'Given a question, convert it to a standalone question. Question: {question} standalone question:'

const tweetPrompt = PromptTemplate.fromTemplate(tweetTemplate)

const tweetChain = tweetPrompt.pipe(model).pipe(new StringOutputParser()).pipe(retriever).pipe(combineDocs)


const punctuationChain = RunnableSequence.from([punctuationPrompt, model, new StringOutputParser()])
const grammarChain = RunnableSequence.from([grammarPrompt, model, new StringOutputParser()])
const translationChain = RunnableSequence.from([translationPrompt, model, new StringOutputParser()])


const standaloneChain = RunnableSequence.from([tweetPrompt, model, new StringOutputParser(),retriever,combineDocs])

const chain2 = RunnableSequence.from([
    {standalone_question: standaloneChain},
    {
        context: prev=>prev,
        question: new RunnablePassthrough()
    },
    answerPrompt,
    model,
    new StringOutputParser()
])



const chain = RunnableSequence.from([
    {
        punctuated_sentence: punctuationChain,
        original_input: new RunnablePassthrough()
    },
    {
        grammatically_correct_sentence: grammarChain,
        language:({ original_input}) => original_input.language
    },
    translationChain
])

// const chain = RunnableSequence.from([
  
//     punctuationPrompt,
//     model,
//     new StringOutputParser(),
//     {punctuated_sentence: prev=>prev},
//     grammarPrompt,
//     model,
//     new StringOutputParser(),
//     {grammatically_correct_sentence: prev=>prev,language:(original_input)=>console.log(original_input)},
//     translationPrompt,
//     model,
//     new StringOutputParser(),
// ])

// const response = await chain.invoke({ question: 'What are the technica requirements for running langchain?, and what are the benefits of using langchain? I have a very old laptop which is not very powerful' })
// const response2 = await retriever.invoke( 'Will scrimba work on my old laptop?')
// console.log(await tweetPrompt.format({ description: 'A product that is a laptop' }))
// console.log(await tweetChain.invoke({ description: 'A product that is a laptop' }))
// console.log(response.content)
const response = await chain.invoke({
     sentence: 'i dont liked mondays bcus its wen d weik STARTS',
     language: 'french',
    
    })

const response2 = await chain2.invoke({
    question: 'what are the technical requirements for running scrimba on my old laptop?'})
console.log(response2)
