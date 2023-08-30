import React, { useEffect, useState } from 'react';
import LineChart from '../../charts/LineChart02Freq';
import Info from '../../utils/Info';

import { tailwindConfig, hexToRGB } from '../../utils/Utils';

function PositiveOverTime(props) {
  const { dateFilter } = props;
  const apiData = props.mongoData;

  const [labels, setLabels] = useState([])
  const [datasetPos, setDatasetPos] = useState([])
  const [datasetNeg, setDatasetNeg] = useState([])

  useEffect(() => {
    let reviews = []

    if (apiData != null) {
      let onesAccumulated = null;
      let zerosAccumulated = null;
      let intervalStartDates = null;

      if (apiData.review_data.length > 0) {
        //Filtrez recenziile care nu au un sentiment valid si apoi le sortez dupa data publicarii
        reviews = apiData.review_data.filter((rev) => rev.sentiment != -1).sort((a, b) => new Date(a.date_published) - new Date(b.date_published))

        const startDate = new Date(reviews[0].date_published);
        const endDate = new Date(reviews[reviews.length - 1].date_published);
        const intervalMs = (endDate - startDate) / 4;
        //pentru a afisa grafic datele, trebuie sa existe minim 3 recenzii intr-un interval de minim 4 zile
        if (reviews.length > 2 && ((endDate - startDate) >= 345600000)) {

          intervalStartDates = [];
          //daca nu am minim 4 puncte de afisat pe grafic, atunci afisez cate un punct per review
          let numberPoints = reviews.length >= 4 ? 4 : reviews.length;
          for (let i = 0; i < numberPoints; i++) {
            const intervalStartDate = new Date(startDate.getTime() + i * intervalMs);
            const year = intervalStartDate.getFullYear();
            const month = String(intervalStartDate.getMonth() + 1).padStart(2, '0');
            const day = String(intervalStartDate.getDate()).padStart(2, '0');
            intervalStartDates.push(`${year}-${month}-${day}`);
          }

          const intervals = reviews.reduce((acc, rev) => {
            let iterData = new Date(rev.date_published)
            const intervalIndex = Math.floor((iterData - startDate) / intervalMs);
            if (intervalIndex >= 0 && intervalIndex < numberPoints) {
              acc[intervalIndex].push(rev);
            }
            return acc;
          }, [[], [], [], []]);

          const onesByInterval = intervals.map(interval =>
            interval.filter(item => item.sentiment === 1)
              .length
          );

          const zerosByInterval = intervals.map(interval =>
            interval.filter(item => item.sentiment === 0)
              .length
          );

          onesAccumulated = onesByInterval.reduce((acc, count) => {
            const lastValue = acc.length > 0 ? acc[acc.length - 1] : 0;
            acc.push(lastValue + count);
            return acc;
          }, []);

          zerosAccumulated = zerosByInterval.reduce((acc, count) => {
            const lastValue = acc.length > 0 ? acc[acc.length - 1] : 0;
            acc.push(lastValue + count);
            return acc;
          }, []);
          
          for (let i = 1; i < intervalStartDates.length; i++) {
            if (intervalStartDates[i] == intervalStartDates[i - 1]) {
              intervalStartDates.splice(0, i)
              onesAccumulated.splice(0, i)
              zerosAccumulated.splice(0, i)
            }
        }
        }
      }
      else {
        onesAccumulated = []
        zerosAccumulated = []
        intervalStartDates = []
      }



      setDatasetPos(onesAccumulated);
      setDatasetNeg(zerosAccumulated);
      setLabels(intervalStartDates);

    }

  }, [apiData]);

  const chartData = {
    labels: labels,
    datasets: [
      // Indigo line
      {
        label: 'Recenzii pozitive',
        data: datasetPos,
        backgroundColor: `rgba(${hexToRGB(tailwindConfig().theme.colors.blue[500])}, 0.08)`,
        borderColor: tailwindConfig().theme.colors.indigo[500],
        fill: true,
        borderWidth: 2,
        tension: 0,
        pointRadius: 3,
        pointHoverRadius: 3,
        pointBackgroundColor: tailwindConfig().theme.colors.indigo[500],
      },
      // Blue line
      {
        label: 'Recenzii negative',
        data: datasetNeg,
        backgroundColor: `rgba(${hexToRGB(tailwindConfig().theme.colors.indigo[500])}, 0.08)`,
        borderColor: tailwindConfig().theme.colors.blue[400],
        fill: true,
        borderWidth: 2,
        tension: 0,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointBackgroundColor: tailwindConfig().theme.colors.blue[400],
      }
    ],
  };

  return (
    <>
      <div className="flex flex-col col-span-full bg-white shadow-lg rounded-sm border border-slate-200">
        <header className="px-5 py-4 border-b border-slate-100 flex items-center">
          <h2 className="text-lg font-semibold text-slate-800">Frecvența polarității recenziilor</h2>
          <Info className="ml-2" containerClassName="min-w-80">
            <div className="text-sm">Sentimentul generat de recenziile oferite de utilizatori</div>
          </Info>
        </header>
        <LineChart data={chartData} width={595} height={248} />
      </div>
    </>
  );
}

export default PositiveOverTime;
