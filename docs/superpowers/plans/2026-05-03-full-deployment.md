# Full Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the Digital India project as a Next.js report site on Vercel with a FastAPI backend on Render, including an LLM-powered data chat.

**Architecture:** Next.js app with MDX report pages (static) + interactive explorer page that calls a FastAPI backend. Backend serves pre-computed data from SQLite and runs LLM-generated code for the chat feature. MongoDB data is flattened into SQLite for deployment.

**Tech Stack:** Next.js 14 (App Router, MDX), FastAPI, SQLite, Plotly.js, react-plotly.js, Anthropic SDK, Tailwind CSS.

---

## Summary of Tasks

1. Migrate MongoDB to SQLite (backend/migrate_mongo.py)
2. FastAPI backend with data endpoints (backend/main.py)
3. LLM chat backend (backend/chat.py, backend/sandbox.py)
4. Render deployment config
5. Next.js frontend scaffold
6. Root layout and navigation
7. Report pages (MDX) with figures
8. Chat component (frontend)
9. Vercel deployment config
10. Deploy both services
11. Write remaining report MDX content

See the full spec at: docs/superpowers/specs/2026-05-03-llm-repl-design.md

## Detailed task breakdown available in conversation context.
