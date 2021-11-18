
def get_manasses_columns():
    return ["procedencia_id", "convenio_id", "usuario_criador_id", "descricao_macroscopia", "descricao_microscopia", "paciente_solicita_exame_mail"]


def get_modesto_columns():
    return ["sigla", "representacao", "malignidade", "atipia"]


def get_romario_columns():
    return ["id_descricao_laudo", "paciente_id", "bairro", "material_devolvido", "numero", "uf"]


def get_gabriel_columns():
    return ['paciente_solicita_exame_site', 'op3031_id', 'tipo_atendimento', 'descricao_conclusao', 'op1314', 'idade/dt. nasc']


def get_vinicius_columns():
    return ['exame_id', 'codigo_barra', 'total_pagar', 'date_created', 'material_recebido', 'data_recepcao']


def get_pedro_columns():
    return ['id_paciente', 'sexo', 'cobrar_de', 'codigo_usuario_convenio', 'dados_clinicos', 'diagnostico_clinico']


def get_yasmin_columns():
    return ['tipo_exame_id', 'paciente_solicita_entrega', 'devolvido', 'local_atendimento', 'peca', 'quantidade_pecas']


def get_rafael_columns():
    return ['data_recepcao', 'data_autorizacao', 'data_entrega', 'data_designado', 'data_prometido', 'data_requisicao', 'data_macroscopia', 'data_conclusao', 'data_liberacao']


def get_walmir_columns():
    return ['carater_atendimento', 'medico_requisitante_id', 'usuario_recepcao_id', 'usuario_conclusao_id', 'auxiliar_macroscopia_id', 'responsavel_macroscopia_id']
