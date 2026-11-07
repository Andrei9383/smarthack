import type { UpdateParameters, Viewport } from "@deck.gl/core";
import { WindParticleLayer } from "maplibre-gl-wind";

const isTexture = (image: unknown) => typeof image === "object" && image !== null;

// maplibre-gl-wind with two fixes for an hourly, country-sized field:
// - a new hour's image only swaps the wind texture, so particles keep flowing instead of all
//   respawning at random spots on every replay step;
// - particles spawn, and are dropped, inside the data bounds, not the whole viewport: outside the
//   field the library freezes them in place, which wastes the particle budget off Romania.
export class SteadyWindLayer extends WindParticleLayer {
  static layerName = "SteadyWindLayer";

  updateState(params: UpdateParameters<this>) {
    const { props, oldProps } = params;
    const hourChanged =
      this.state.initialized && props.image !== oldProps.image && isTexture(props.image) && isTexture(oldProps.image);
    if (!hourChanged) {
      super.updateState(params);
      return;
    }
    // Hide the image change from the parent so it doesn't rebuild the particle buffers.
    const sameImage = Object.create(oldProps, { image: { value: props.image } });
    super.updateState({ ...params, oldProps: sameImage });
    this.state.texture = props.image as typeof this.state.texture;
  }

  _runTransformFeedback() {
    const viewport = this.context.viewport;
    const [west, south, east, north] = this.props.bounds;
    // The parent reads the spawn area from viewport.getBounds(); clip it to the field for one step.
    this.context.viewport = Object.create(viewport, {
      getBounds: {
        value: () => {
          const [minLng, minLat, maxLng, maxLat] = viewport.getBounds();
          return [Math.max(minLng, west), Math.max(minLat, south), Math.min(maxLng, east), Math.min(maxLat, north)];
        },
      },
    }) as Viewport;
    try {
      super._runTransformFeedback();
    } finally {
      this.context.viewport = viewport;
    }
  }
}
