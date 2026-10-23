import { useQuery, useMutation } from '@tanstack/react-query';
import {
  MOCK_PATIENTS,
  MOCK_PATIENT_DETAILS,
  MOCK_PREDICTIONS,
  MOCK_HISTORIES,
  MOCK_EXPLANATIONS,
  MODEL_METADATA,
  computeWhatIf,
} from './mockData';
import type {
  PatientSummary,
  PatientDetail,
  PredictResponse,
  CGMDataPoint,
  ExplainResponse,
  ModelMetadata,
  WhatIfResponse,
  WhatIfRequest,
} from '@/types/schema';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    ...init,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ─── Patient List ──────────────────────────────────────────────────────────

export function usePatients() {
  return useQuery<PatientSummary[]>({
    queryKey: ['patients'],
    queryFn: async () => {
      if (USE_MOCK) { await delay(400); return MOCK_PATIENTS; }
      return apiFetch<PatientSummary[]>('/patients');
    },
    staleTime: 60 * 1000,
    refetchInterval: 60 * 1000,
  });
}

// ─── Patient Detail ────────────────────────────────────────────────────────

export function usePatient(id: string) {
  return useQuery<PatientDetail>({
    queryKey: ['patient', id],
    queryFn: async () => {
      if (USE_MOCK) {
        await delay(250);
        const p = MOCK_PATIENT_DETAILS[id];
        if (!p) throw new Error(`Patient ${id} not found`);
        return p;
      }
      return apiFetch<PatientDetail>(`/patients/${id}`);
    },
    enabled: !!id,
    staleTime: 2 * 60 * 1000,
  });
}

// ─── Prediction ────────────────────────────────────────────────────────────

export function usePrediction(id: string) {
  return useQuery<PredictResponse | null>({
    queryKey: ['prediction', id],
    queryFn: async () => {
      if (USE_MOCK) { await delay(300); return MOCK_PREDICTIONS[id] ?? null; }
      return apiFetch<PredictResponse>(`/patients/${id}/predict`);
    },
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
  });
}

// ─── CGM History ───────────────────────────────────────────────────────────

export function useCGMHistory(id: string) {
  return useQuery<CGMDataPoint[]>({
    queryKey: ['cgm', id],
    queryFn: async () => {
      if (USE_MOCK) { await delay(200); return MOCK_HISTORIES[id] ?? []; }
      return apiFetch<CGMDataPoint[]>(`/patients/${id}/history`);
    },
    enabled: !!id,
    staleTime: 55 * 1000,
    refetchInterval: 60 * 1000,
  });
}

// ─── Explanation ───────────────────────────────────────────────────────────

export function useExplanation(id: string) {
  return useQuery<ExplainResponse | null>({
    queryKey: ['explain', id],
    queryFn: async () => {
      if (USE_MOCK) { await delay(350); return MOCK_EXPLANATIONS[id] ?? null; }
      return apiFetch<ExplainResponse>(`/patients/${id}/explain`);
    },
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
  });
}

// ─── Model Info ────────────────────────────────────────────────────────────

export function useModelInfo() {
  return useQuery<ModelMetadata>({
    queryKey: ['model-info'],
    queryFn: async () => {
      if (USE_MOCK) { await delay(200); return MODEL_METADATA; }
      return apiFetch<ModelMetadata>('/model/info');
    },
    staleTime: 30 * 60 * 1000,
  });
}

// ─── What-If Mutation ──────────────────────────────────────────────────────

export function useWhatIf(patientId: string) {
  return useMutation<WhatIfResponse, Error, WhatIfRequest>({
    mutationFn: async (req: WhatIfRequest) => {
      if (USE_MOCK) {
        await delay(500);
        return computeWhatIf(patientId, req.carbs_g, req.walk_duration_min, req.stress_level);
      }
      return apiFetch<WhatIfResponse>(`/patients/${patientId}/whatif`, {
        method: 'POST',
        body: JSON.stringify(req),
      });
    },
  });
}
