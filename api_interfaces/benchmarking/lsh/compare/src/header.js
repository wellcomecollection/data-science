import React, { Component } from "react";
import WellcomeImage from "./wellcome_image";

class Header extends Component {
  render() {
    return (
      <section className="bg-light-gray pa3 ph5-ns">
        <h1 className="mt0">Subjective similarity assessment</h1>
        <p className="lh-copy measure">
          Choose the set of images which looks most similar to this query image
        </p>
        <div className="w-75 center">
          {<WellcomeImage id={this.props.query_id}></WellcomeImage>}
        </div>
      </section>
    );
  }
}

export default Header;
