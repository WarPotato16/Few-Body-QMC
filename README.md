# Quantum Monte Carlo (QMC) Research Project

## Overview

This repository contains code, notes, and supporting material for research on Quantum Monte Carlo (QMC) methods applied to interacting quantum many-body systems.

The goal is to study strongly correlated systems using stochastic sampling techniques 
---

## What is Quantum Monte Carlo?

Quantum Monte Carlo (QMC) refers to a class of numerical methods that use stochastic sampling to evaluate quantum mechanical observables.

The central idea is to rewrite quantum expectation values as statistical averages over configurations sampled from a probability distribution.

In many cases, QMC provides a non-perturbative and systematically improvable approach to interacting quantum systems.

---

## Key Idea

A typical quantum expectation value has the form:

$$
\langle \mathcal{O} \rangle = \frac{\langle \psi | \mathcal{O} | \psi \rangle}{\langle \psi | \psi \rangle}
$$

In QMC, this is reformulated as:

- a sum or integral over configurations \( C \)
- weighted by a probability distribution \( P(C) \)
- estimated via random sampling

$$
\langle \mathcal{O} \rangle \approx \frac{1}{N} \sum_{i=1}^N \mathcal{O}(C_i)
$$

---

## Methods Included

This project may include implementations of:

- Path-integral Monte Carlo (PIMC)
- Variants for fermionic systems (sign problem considerations)

---

## Applications

We apply QMC methods to study:

- Few-body and many-body fermionic systems
- Lattice discretizations of quantum field theories
- Effective field theory (EFT) based Hamiltonians

---
