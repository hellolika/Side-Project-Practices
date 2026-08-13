using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class TimeTableResponse
    {
        private TimeSpan _clientOffSet { get; set; }

        [JsonProperty("WorkDate")]
        [JsonConverter(typeof(CustomDateTimeConverter))]
        public DateTime WorkDate { get; set; }

        [JsonProperty("ClockIn")]
        [JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan? ClockIn { get; set; }

        [JsonProperty("ClockOut")]
        [JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan? ClockOut { get; set; }

        [JsonProperty("ReClockIn")]
        public ReClockInfo ReClockIn { get; set; }

        [JsonProperty("ReClockOut")]
        public ReClockInfo ReClockOut { get; set; }

        [JsonProperty("LeaveInfo")]
        public LeaveTypeInfo LeaveInfo { get; set; }

        public TimeTableResponse()
        { }

        public TimeTableResponse(DateTime workDate, TimeSpan clientOffSet)
        {
            WorkDate = workDate;
            _clientOffSet = clientOffSet;
        }

        public void FillAttendance(Attendance attendance)
        {
            ClockIn = attendance.ClockIn?.Add(_clientOffSet);
            ClockOut = attendance.ClockOut?.Add(_clientOffSet);
        }

        public void FillReClockRecord(ReClockRecord reClock)
        {
            if (reClock.IsClockIn)
            {
                ReClockIn = new ReClockInfo(reClock.Time, reClock.Status);
            }
            else
            {
                ReClockOut = new ReClockInfo(reClock.Time, reClock.Status);
            }
        }

        public void FillLeaveRecord(TakeLeaveRecord leaveRecord)
        {
            LeaveInfo = new LeaveTypeInfo(leaveRecord);
        }
    }

    public class ReClockInfo
    {
        [JsonProperty("Time")]
        //[JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan Time { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }

        public ReClockInfo(TimeSpan time, StatusEnum status)
        {
            Time = time;
            Status = status;
        }
    }

    public class LeaveTypeInfo
    {
        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }
        
        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }

        [JsonProperty("IsCancel")]
        public bool IsCancel { get; set; }

        [JsonProperty("NumberOfDay")]
        public decimal NumberOfDay { get; set; }

        public LeaveTypeInfo(TakeLeaveRecord leaveRecord)
        {
            LeaveId = leaveRecord.LeaveId;
            LeaveType = leaveRecord.LeaveType;
            Status = leaveRecord.Status;
            IsCancel = leaveRecord.IsCancel;
            NumberOfDay = leaveRecord.NumberOfDay;
        }
    }
}