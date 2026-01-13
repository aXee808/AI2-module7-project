{{/*
_helpers.tpl - Fonctions helper reutilisables
Ces fonctions simplifient l'ecriture des templates
*/}}

{{/*
Nom du chart
*/}}
{{- define "stock-prediction.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Nom complet de la release (release-name + chart-name)
Exemple: my-release-stock-prediction
*/}}
{{- define "stock-prediction.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Labels standards Kubernetes (pour tous les objets)
*/}}
{{- define "stock-prediction.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/name: {{ include "stock-prediction.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Labels de selection (pour les selectors de Service et Deployment)
*/}}
{{- define "stock-prediction.selectorLabels" -}}
app.kubernetes.io/name: {{ include "stock-prediction.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Nom de l'image complete (repository:tag)
*/}}
{{- define "stock-prediction.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
