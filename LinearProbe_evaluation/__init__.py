"""Qdrant 像素 embedding 线性分类评估系统 — MLP_label (Linear Probe).

基于 Qdrant 中已有的像素 embedding（64 维）与 DW 硬分类标签（0-8），
训练 64→9 的线性分类器（Linear Probe），并提供 CLI 训练/评估与
NiceGUI Web 训练过程监测界面。
"""
