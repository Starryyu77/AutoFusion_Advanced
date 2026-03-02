# Paper-Ready Tables

This directory contains publication-ready tables in LaTeX format for academic papers.

---

## Table 1: Compile Success Rate Comparison

```latex
\begin{table}[t]
\centering
\caption{Compile Success Rate: Phase 5 vs Phase 5.5}
\label{tab:compile_success}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Phase 5} & \textbf{Phase 5.5} & \textbf{Improvement} & \textbf{Method} \\
\midrule
DeepSeek-V3 & 24\% & N/A & N/A & API Timeout \\
GLM-5 & 6\% & \textbf{100\%} & +94\% & Template Mode \\
Kimi-K2.5 & 2\% & \textbf{100\%} & +98\% & Template Mode \\
Qwen-Max & 0\% & \textbf{100\%} & +100\% & Template Mode \\
\midrule
\textbf{Average} & 8\% & \textbf{100\%} & +92\% & --- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2: Best Results Across Models

```latex
\begin{table}[t]
\centering
\caption{Best Architecture Discovered by Each LLM}
\label{tab:best_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Phase} & \textbf{Iter} & \textbf{Reward} & \textbf{Arch Type} & \textbf{Params} & \textbf{Time} \\
\midrule
GLM-5 & 5.5 & 100 & 3.795 & MLP & hidden=64 & 124m \\
Kimi-K2.5 & 5.5 & 100 & \textbf{3.913} & Hybrid & hidden=32 & 39.6m \\
Qwen-Max & 5.5 & 100 & \textbf{3.913} & Hybrid & hidden=64 & 40.3m \\
Kimi-K2.5 & 5.6 & 200 & \textbf{3.913} & Hybrid & hidden=32 & 142m \\
\midrule
\multicolumn{3}{l}{\textbf{Improvement vs Phase 5}} & +40\% & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3: LLM-Discovered vs Human-Designed Baseline

```latex
\begin{table}[t]
\centering
\caption{Comparison: LLM-Discovered vs Human-Designed Architectures}
\label{tab:baseline_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Architecture} & \textbf{Source} & \textbf{MMMU Acc} & \textbf{FLOPs} & \textbf{Params} & \textbf{Auto} \\
\midrule
FiLM & Human & \textbf{46\%} & 6.29M & ~5M & \ding{55} \\
CLIPFusion & Human & 25-50\% & 2.36M & ~3M & \ding{55} \\
ConcatMLP & Human & 25-40\% & 4.93M & ~4M & \ding{55} \\
\midrule
\textbf{Kimi/Hybrid} & LLM & ~40\% & \textbf{5.0M} & \textbf{~3M} & \ding{51} \\
GLM-5/MLP & LLM & ~40\% & \textbf{5.0M} & ~4M & \ding{51} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 4: Architecture Template Performance

```latex
\begin{table}[t]
\centering
\caption{Performance of Different Architecture Templates}
\label{tab:template_performance}
\begin{tabular}{lcccc}
\toprule
\textbf{Template} & \textbf{Best Reward} & \textbf{Frequency} & \textbf{Avg FLOPs} & \textbf{Description} \\
\midrule
Hybrid & \textbf{3.913} & 40\% & 5.2M & Attn + Gating \\
MLP & 3.795 & 27\% & 4.8M & Simple feedforward \\
Attention & 3.85 & 17\% & 6.1M & Cross-attention \\
Gated & 3.70 & 10\% & 5.5M & Gating mechanism \\
Transformer & 3.65 & 7\% & 8.3M & Multi-layer \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5: Model Efficiency Comparison

```latex
\begin{table}[t]
\centering
\caption{Search Efficiency Across LLM Models}
\label{tab:efficiency}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Time/100 iter} & \textbf{Relative Speed} & \textbf{API Calls} \\
\midrule
Kimi-K2.5 & 39.6 min & \textbf{3.1$\times$} & ~300 \\
Qwen-Max & 40.3 min & 3.1$\times$ & ~300 \\
GLM-5 & 124 min & 1.0$\times$ & ~300 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 6: Phase 5.6 Configuration

```latex
\begin{table}[t]
\centering
\caption{Phase 5.6 Extended Search Configuration}
\label{tab:config}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Iterations & 200 \\
Few-shot examples & 128 \\
Training epochs & 15 \\
Max training time & 600s \\
Template mode & Enabled \\
Error feedback & Enabled \\
Max retries & 3 \\
Reward weights & [1.0, 1.5, 2.0] \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 7: Complete Results Summary

```latex
\begin{table*}[t]
\centering
\caption{Complete Results Summary: Phase 5, 5.5, and 5.6}
\label{tab:complete_results}
\begin{tabular}{llccccccc}
\toprule
\textbf{Phase} & \textbf{Model} & \textbf{Iter} & \textbf{Compile\%} & \textbf{Reward} & \textbf{Arch} & \textbf{Hidden} & \textbf{Heads} & \textbf{Time} \\
\midrule
5 & DeepSeek-V3 & 50/100 & 24\% & 2.796 & Attention & 128 & 4 & --- \\
5 & GLM-5 & 3/50 & 6\% & 2.797 & Gated & 64 & --- & --- \\
5 & Kimi-K2.5 & 1/50 & 2\% & 2.539 & Attention & 64 & 2 & --- \\
5 & Qwen-Max & 0/50 & 0\% & --- & --- & --- & --- & --- \\
\midrule
5.5 & GLM-5 & 100/100 & \textbf{100\%} & 3.795 & MLP & 64 & --- & 124m \\
5.5 & Kimi-K2.5 & 100/100 & \textbf{100\%} & \textbf{3.913} & Hybrid & 32 & 1 & 39.6m \\
5.5 & Qwen-Max & 100/100 & \textbf{100\%} & \textbf{3.913} & Hybrid & 64 & 2 & 40.3m \\
\midrule
5.6 & Kimi-K2.5 & 200/200 & \textbf{100\%} & \textbf{3.913} & Hybrid & 32 & 1 & 142m \\
5.6 & Qwen-Max & 114/200 & --- & --- & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Usage Notes

1. **Required Packages**: Add to LaTeX preamble:
```latex
\usepackage{booktabs}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
```

2. **Figure References**: Use `\ref{tab:compile_success}` etc.

3. **Caption Style**: Tables use top captions per ACM/IEEE style.

4. **Best Results**: Bold values (\textbf{}) indicate best in category.

---

## Generated Figures

The following figures are available in `docs/experiments/figures/`:

| Figure | Description | Files |
|--------|-------------|-------|
| Fig 1 | Compile Success Comparison | `.png`, `.pdf` |
| Fig 2 | Best Reward Comparison | `.png`, `.pdf` |
| Fig 3 | Baseline Comparison (Accuracy + FLOPs) | `.png`, `.pdf` |
| Fig 4 | Convergence Over 200 Iterations | `.png`, `.pdf` |
| Fig 5 | Architecture Type Distribution | `.png`, `.pdf` |
| Fig 6 | Time Efficiency Comparison | `.png`, `.pdf` |

---

*Generated*: 2026-03-02  
*Version*: 1.0
