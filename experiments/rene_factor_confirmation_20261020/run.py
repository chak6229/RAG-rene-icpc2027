from . import inference, analysis, audit, report
from .common import OUT


def main():
    inference.generate()
    inference.rank()
    if not (OUT/'PREOPEN_AUDIT.json').exists():audit.preopen()
    if not (OUT/'RESULTS.json').exists():analysis.main()
    if not (OUT/'EFFECT_AUDIT.json').exists():audit.effect()
    if not (OUT/'MODEL_AUDIT.json').exists():audit.models()
    if not (OUT/'FINAL_ACCEPTANCE.json').exists():report.main()


if __name__=='__main__':main()
