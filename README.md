# C3-Bench: A Context-Aware Change Captioning Benchmark
This repository represents the official implementation of the paper titled "C3-Bench: A Context-Aware Change Captioning Benchmark (ECCV 2026)". 

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-929292)](LICENSE)
[![Website](https://img.shields.io/badge/Website-ECCV_2026-blue)](https://eccv.ecva.net/)
[![ECCV 2026 Paper](https://img.shields.io/badge/ECCV%202026-Paper-blue)](https://arxiv.org/abs/XXXX.XXXXX)

<p align="center"> <a href="https://1124jaewookim.github.io/"><strong>Jaewoo Kim</strong></a> · <a href="https://hbk08101.github.io/"><strong>Hyeongbeom Kim</strong></a> · <a href="https://uehwan.github.io/people/Ue-Hwan-Kim/"><strong>Uehwan Kim</strong></a> <br> <strong>ECCV 2026</strong> </p> <div align='center'> <br><img src="image/teaser-1.png" width=100%> <br><strong>Overview of C3-Bench.</strong> The examples are from each context in C3-Bench. </div> <br><br>

<table>
<tr>
<td width="55%" valign="top">

<h2>💡 Problem Formulation</h2>

<p>
Change Captioning aims to describe the changes between two images.
</p>

<p>
However, <i>what counts as change</i> is totally context-dependent.
For example, when one is asked to describe the change between the given image pair,
what might first come to mind is <b>"in which context?"</b>.
</p>

<p>
<i>the snow has covered the ground</i> for weather analysis, or
<i>a train has appeared on the left side of the tracks</i> for railway surveillance,
with weather differences treated as pseudo-changes.
</p>

</td>

<td width="45%" align="center">

<img src="image/hook (1)-1.png" width="100%">

<br><br>

<b>What has changed? (Motivation)</b><br>
Without context, this generic question can admit multiple logically valid descriptions.

</td>
</tr>
</table>
