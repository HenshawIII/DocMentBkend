// import { ChatOpenAI } from "@langchain/openai"
import { PromptTemplate } from "@langchain/core/prompts"
import dotenv from "dotenv"
import { vectorStoreRetrieval,model,embeddings } from "./index.js"
import { StringOutputParser } from "@langchain/core/output_parsers"
import { RunnableSequence ,RunnablePassthrough} from "@langchain/core/runnables"
import { convHistory } from "./server.js"

dotenv.config()


function combineDocs(docs) {
    return docs.map(doc => doc.pageContent).join("\n\n")
} 



const retriever = vectorStoreRetrieval.asRetriever({
    filter:{
        user_id:"Sharon",
        document_id:"c8795065-485f-43b4-962d-9df829b4e92c"
    }
})

const standaloneQuestionTemplate = 'Given a question and conversation history if any, convert the question to a standalone question.conversation history: {convHistory} question: {question} standalone question:'
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given questions based on the context provided and the conversation history if any. Try to find the answer in the context.If the question is related to the conversation history, use the conversation history to answer the question. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@Hictech.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
conversation history: {convHistory}
question: {question}
answer: `
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const standaloneQuestionChain = standaloneQuestionPrompt
    .pipe(model)
    .pipe(new StringOutputParser())
    
const retrieverChain = RunnableSequence.from([
    prevResult => prevResult.standalone_question,
    retriever,
    combineDocs
])
const answerChain = answerPrompt
    .pipe(model)
    .pipe(new StringOutputParser())

export const chain = RunnableSequence.from([
    {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough()
    },
    {
        context: retrieverChain,
        question: ({ original_input }) => original_input.question,
        convHistory: ({ original_input }) => original_input.convHistory
    },
    answerChain
]) 

// const response = await chain.invoke({
//     question: 'will I interact with other students?'
// })

// console.log(response)





