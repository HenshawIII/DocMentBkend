import express from 'express';
// import { chain } from './chat2.js';
import dotenv from 'dotenv';
import multer from 'multer';
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai"
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import {TextLoader} from "langchain/document_loaders/fs/text"
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import cors from 'cors';
import session from 'express-session';
import { supabaseClient } from './index.js';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { model } from './index.js';
import { vectorStoreRetrieval } from './index.js';
// import { MemoryStore } from 'express-session';
import cookieParser from 'cookie-parser';
import sqlite from 'better-sqlite3';
import CreateSqliteStore from "better-sqlite3-session-store";

dotenv.config();

const app = express();
app.use(cookieParser())
app.use(express.json());
app.use(express.static('.'));
app.use(cors({
    origin: "https://docu-mentor-murex.vercel.app", // Add your frontend URLs
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// const SqliteStore = require("better-sqlite3-session-store")(session)
const SqliteStore = CreateSqliteStore(session)
const db = new sqlite("sessions.db", { verbose: console.log });
// Session middleware
app.use(session({
    secret: process.env.SESSION_SECRET || 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    store: new SqliteStore({
        client: db,
        expired:{
            clear:true,
            intervalMs:900000
        } // 24 hours
    }),
    cookie: { 
        secure:true,// set to true if using https
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        httpOnly: true,
        sameSite: 'none'
    }
}));

const storage = multer.memoryStorage();
const upload = multer({ storage });

// Remove the global convHistory
// export const convHistory =[]

function combineDocs(docs) {
    return docs.map(doc => doc.pageContent).join("\n\n")
} 

// app.post('/api/chat', async (req, res) => {
//     try {
//         const { question } = req.body;   
//         const response = await chain.invoke({ question, convHistory });
//         convHistory.push({ Human: question, AI: response });
//     //    console.log(convHistory)
//         res.json({ response });
//     } catch (error) {
//         console.error('Error:', error);
//         res.status(500).json({ error: 'Internal server error' });
//     }
// });




app.post('/api/user/add', upload.single('file'), async (req, res) => {
    try {
        const {user_id} = req.body
        const fileBuffer = req.file.buffer
        const fileName = req.file.originalname
        const fileMimeType = req.file.mimetype
        const documentId = crypto.randomUUID();
        
        if(fileBuffer){
            // console.log(fileBuffer)
            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 500,
                chunkOverlap: 50,
            });
            
            let loader;
            let docs;
            
            // Check file type and use appropriate loader
            if (fileMimeType === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
                // Handle PDF files
                const blob = new Blob([fileBuffer], { type: 'application/pdf' });
                loader = new PDFLoader(blob);
            } else if (fileMimeType === 'text/plain' || fileName.toLowerCase().endsWith('.txt')) {
                // Handle TXT files
                const textContent = fileBuffer.toString('utf-8');
                loader = new TextLoader(textContent);
            } else {
                return res.status(400).json({ error: 'Unsupported file type. Please upload a PDF or TXT file.' });
            }
            
            docs = await loader.load();
            const chunks = await splitter.splitDocuments(docs);
            
            // optional, for grouping
            const chunksWithMetadata = chunks.map((chunk, index) => ({
              pageContent: chunk.pageContent,
              metadata: {
                user_id: user_id,
                document_id: documentId,
                
              }
            }));
            
            const vectorStore = await SupabaseVectorStore.fromDocuments(chunksWithMetadata, new OpenAIEmbeddings({
                openAIApiKey: process.env.OPENAI_API_KEY,
                modelName: "text-embedding-3-small",
            }), {
                client: supabaseClient,
                tableName: "documents",
            });
            // console.log(chunksWithMetadata)
        }
        res.status(200).json({ message: 'Data received successfully' ,documentId});
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post("/api/user/chat", async (req, res) => {
    try {
        const {user_id, document_id, question} = req.body;
        
        // console.log('Session ID:', req.sessionID);
        // console.log('Session exists:', !!req.session);
        // console.log('Session data:', req.session);
        
        // Initialize conversation history for this session if it doesn't exist
        if (!req.session.convHistory) {
            req.session.convHistory = {};
            // console.log('Initialized new convHistory object');
        }
        
        // Initialize conversation history for this specific document if it doesn't exist
        if (!req.session.convHistory[document_id]) {
            req.session.convHistory[document_id] = [];
            // console.log(`Initialized new conversation history for document ${document_id}`);
        }
        
        // console.log(`Current conversation history for ${document_id}:`, req.session.convHistory[document_id]);
        
        const retriever = vectorStoreRetrieval.asRetriever({
            filter:{
                user_id:user_id,
                document_id:document_id
            }
        })
        
        const standaloneQuestionTemplate = 'Given a question and conversation history if any, convert the question to a standalone question.conversation history: {convHistory} question: {question} standalone question:'
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)
        
        const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given questions based on the context provided and the conversation history if any. Try to find the answer in the context.If the question is related to the conversation history, use the conversation history to answer the question. If you really don't know the answer, say "I'm sorry, I don't know the answer to that.". Don't try to make up an answer. Always speak as if you were chatting to a student.
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
        
         const chain = RunnableSequence.from([
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
        
        const response = await chain.invoke({ question, convHistory: req.session.convHistory[document_id] });
        req.session.convHistory[document_id].push({ Human: question, AI: response });
        
        // Save the session explicitly
        req.session.save((err) => {
            if (err) {
                // console.error('Error saving session:', err);
            } else {
                // console.log('Session saved successfully');
                // console.log('Updated session data:', req.session);
            }
        });
        
        // console.log(`This is the conversation history for ${document_id}`,req.session.convHistory[document_id])
        res.status(200).json({ response });
    }catch(error){
        console.error('Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
})

// Debug route to check session state
app.get("/api/debug/session", (req, res) => {
    console.log('Session ID:', req.sessionID);
    console.log('Session data:', req.session);
    res.json({ 
        sessionID: req.sessionID,
        convHistory: req.session.convHistory || {},
        sessionData: req.session
    });
});

// Simple test route to verify session persistence
app.post("/api/test-session", (req, res) => {
    if (!req.session.testCount) {
        req.session.testCount = 0;
    }
    req.session.testCount++;
    
    req.session.save((err) => {
        if (err) {
            console.error('Error saving test session:', err);
            res.status(500).json({ error: 'Session save failed' });
        } else {
            res.json({ 
                message: 'Session test successful',
                testCount: req.session.testCount,
                sessionID: req.sessionID
            });
        }
    });
});

const PORT = process.env.PORT || 3300;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});


