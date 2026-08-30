# AI Procurement CRM

An AI-powered Procurement Intelligence and CRM platform designed to automate company enrichment, supplier intelligence, contact discovery, and procurement-related research.

The system combines web search, website scraping, LLM-based information extraction, duplicate detection, entity resolution, and structured database management to create reliable and enriched company records.

---

## Overview

Traditional procurement CRMs often rely on manually maintained company and supplier information. This project aims to automate that process using specialized AI agents.

The platform can:

- Discover and validate official company websites
- Enrich company profiles using trusted web sources
- Extract structured company intelligence using Gemini
- Identify procurement and supplier-related information
- Discover relevant procurement contacts
- Detect duplicate company records
- Automatically select the most enriched company as the master record
- Merge duplicate records while preserving available information
- Update related foreign-key references
- Maintain merge logs for traceability
- Coordinate multiple AI agents through a centralized orchestrator

---

## 🧠 System Architecture

```text
                         AI PROCUREMENT CRM
                                │
                                ▼
                         ┌──────────────┐
                         │  Orchestrator│
                         └───────┬──────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      Company Agent        Contact Agent       Supplier Agent
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                         Intelligence Layer
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
         Web Search         Web Scraping          Gemini
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                         Database / Supabase
                                 │
                                 ▼
                       Golden Company Record
