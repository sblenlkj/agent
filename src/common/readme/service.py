from __future__ import annotations

from loguru import logger

from src.common.io.models import InputBundle
from src.common.llm_client import LLMClient
from src.common.readme.merger import ReadmeBundleMerger
from src.common.readme.models import ReadmeParseArtifacts, ReadmeParseResponse
from src.common.readme.parser import ReadmeParser


class ReadmeService:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        parser: ReadmeParser | None = None,
        merger: ReadmeBundleMerger | None = None,
    ) -> None:
        self._merger = merger or ReadmeBundleMerger()
        self._parser = parser or ReadmeParser(
            llm_client=llm_client,
            merger=self._merger,
        )

    def parse(self, bundle: InputBundle) -> ReadmeParseResponse:
        logger.info("ReadmeService.parse started")
        response = self._parser.parse(bundle)
        logger.info("ReadmeService.parse finished")
        return response

    def parse_with_artifacts(self, bundle: InputBundle) -> ReadmeParseArtifacts:
        logger.info("ReadmeService.parse_with_artifacts started")
        artifacts = self._parser.parse_with_artifacts(bundle)
        logger.info("ReadmeService.parse_with_artifacts finished")
        return artifacts

    def enrich_bundle(self, bundle: InputBundle) -> InputBundle:
        logger.info("ReadmeService.enrich_bundle started")
        artifacts = self._parser.parse_with_artifacts(bundle)
        enriched_bundle = self._merger.merge(
            bundle=bundle,
            response=artifacts.parsed_response,
        )
        logger.info("ReadmeService.enrich_bundle finished")
        return enriched_bundle

    def parse_and_enrich(
        self,
        bundle: InputBundle,
    ) -> tuple[ReadmeParseArtifacts, InputBundle]:
        logger.info("ReadmeService.parse_and_enrich started")
        artifacts = self._parser.parse_with_artifacts(bundle)
        enriched_bundle = self._merger.merge(
            bundle=bundle,
            response=artifacts.parsed_response,
        )
        logger.info("ReadmeService.parse_and_enrich finished")
        return artifacts, enriched_bundle