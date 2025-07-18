import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import fs from "fs"
import dotenv from "dotenv"
import { createClient } from "@supabase/supabase-js"
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai"
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import {TextLoader} from "langchain/document_loaders/fs/text"
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf"
// import {puppeteerWebBaseLoader} from "@langchain/community/document_loaders/web/puppeteer"

dotenv.config()

const supabaseUrl = process.env.SUPABASE_URL
const supabaseKey = process.env.SUPABASE_APIKEY
export const supabaseClient = createClient(supabaseUrl, supabaseKey)
export const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY
})


    // const splitter = new RecursiveCharacterTextSplitter({
    //     chunkSize: 500,
    //     chunkOverlap: 50,
    // });
    // const text = fs.readFileSync(path.resolve("scrimba.txt"), "utf-8");

    // const loader = new TextLoader("scrimba.txt")
    // const docs = await loader.load()

    // const pdfLoader = new PDFLoader("rksfac.pdf")
    // const pdfDocs = await pdfLoader.load()

    // console.log(docs)
    // const pdfChunks = await splitter.splitDocuments(pdfDocs)
    // const chunks = await splitter.splitDocuments(docs)
    // console.log(pdfChunks);

    // const vectorStore = await SupabaseVectorStore.fromDocuments(pdfChunks, new OpenAIEmbeddings({
    //     openAIApiKey: process.env.OPENAI_API_KEY,
    //     modelName: "text-embedding-3-small",
    // }), {
    //     client: supabaseClient,
    //     tableName: "documents",
    // })
    // console.log("done")
   export const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        modelName: "text-embedding-3-small",
    })
  export const vectorStoreRetrieval = new SupabaseVectorStore(embeddings, {
        client: supabaseClient,
        tableName: "documents",
        queryName: "match_documents",
    })

    // console.log(vectorStore)









