import App from "next/app";
import Header from "../components/Header";
const fontFamily = "Gadget, sans-serif";

class AppContainer extends App {
  render() {
    const { Component, pageProps } = this.props;
    return (
      <div
        style={{
          fontFamily
        }}
      >
        <Header title={"Labs"} />
        <div
          style={{
            maxWidth: "600px",
            margin: "0 auto"
          }}
        >
          <Component {...pageProps} />
        </div>
      </div>
    );
  }
}

export default AppContainer;
