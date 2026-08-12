using HRMS.Enum;
using HRMS.Filters;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;


namespace HRMS.Controllers
{
    [Route("/api/[controller]")]
    [ServiceFilter(typeof(AuthenticateJwt))]
    [ApiController]
    public class MemberController : ControllerBase
    {
        private readonly IMemberService _memberService;

        public MemberController(IMemberService memberService)
        {
            _memberService = memberService;
        }
        
        [HasPermission(PermissionEnum.CanDoClock)]
        [HttpPost("DoClock")]
        public ApiBaseResponse<string> DoClock(DoClockRequest request)
        {
            return _memberService.DoClock(request);
        }
        
        [HasPermission(PermissionEnum.CanCheckClockStatus)]
        [HttpPost("CheckClockStatus")]
        public ApiBaseResponse<ClockStatusResponse> CheckClockStatus(ClockStatusRequest request)
        {
            return _memberService.CheckClockStatus(request);
        }
        
        [HasPermission(PermissionEnum.CanDoReClock)]
        [HttpPost("DoReClock")]
        public ApiBaseResponse<SubmitFormBaseResponse> DoReClock(DoReClockRequest request)
        {
            return _memberService.DoReclock(request);
        }
        
        [HasPermission(PermissionEnum.CanTakeLeave)]
        [HttpPost("TakeLeave")]
        public ApiBaseResponse<SubmitFormBaseResponse>TakeLeave(TakeLeaveRequest request)
        {
            return  _memberService.TakeLeave(request);
        }
        
        [HasPermission(PermissionEnum.CanCancelLeave)]
        [HttpPost("CancelLeave")]
        public ApiBaseResponse<string> CancelLeave(CancelLeaveRequest request)
        {
            return _memberService.CancelLeave(request);
        }
        
        [HasPermission(PermissionEnum.CanGetTimeTable)]
        [HttpPost("GetTimeTable")]
        public ApiBaseResponse<List<TimeTableResponse>> GetTimeTable(TimeTableRequest request)
        {
            return _memberService.GetTimeTable(request);
        }

        [HasPermission(PermissionEnum.CanGetLeaveAmount)]
        [HttpGet("GetLeaveAmount")]
        public ApiBaseResponse<List<LeaveAmountResponse>> GetLeaveAmount()
        {
            return _memberService.GetMemberLeaveAmount();
        }

        [HasPermission(PermissionEnum.CanGetLeaveAmount)]
        [HttpPost("v2/GetLeaveAmount")]
        public ApiBaseResponse<List<LeaveAmountResponseV2>> GetLeaveAmountV2(ByMemberIdRequest request)
        {
            request.CheckModel();
            return _memberService.GetMemberLeaveAmountV2(request.MemberId);
        }

        [HasPermission(PermissionEnum.CanGetAllRequestForms)]
        [HttpPost("GetAllRequestForms")]
        public ApiBaseResponse<AllRequestFormsResponse> GetAllRequestForms(BaseRequest request)
        {
            return _memberService.GetAllRequestForms(request);
        }

        [HasPermission(PermissionEnum.CanGetProfile)]
        [HttpGet("GetProfile")]
        public ApiBaseResponse<MemberProfile> GetProfile()
        {
            return _memberService.GetProfile();
        }
        
        [HasPermission(PermissionEnum.CanGetProfile)]
        [HttpPost("v2/GetProfile")]
        public ApiBaseResponse<MemberProfile> GetProfileV2(GetProfileRequest request)
        {
            request.CheckModel();
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            // var canSeeMemberSalary = jwtData.Permissions.Contains(Convert.ToInt32(PermissionEnum.CanSeeMemberSalary).ToString());
            return _memberService.GetProfile(memberId: request.MemberId, jwtData.IsAdmin(), jwtData.Id);
        }

        [HasPermission(PermissionEnum.CanUpdateProfile)]
        [HttpPost("UpdateProfile")]
        public ApiBaseResponse<int> UpdateProfile(UpdateProfileRequest request)
        {
            return _memberService.UpdateProfile(request);
        }
        
        [HasPermission(PermissionEnum.CanChangePassword)]
        [HttpPost("ChangePassWord")]
        public ApiBaseResponse<string> ChangePassword(ChangePasswordRequest request)
        {
            return _memberService.ChangePassword(request);
        }
        
        [HasPermission(PermissionEnum.CanGetTransfers)]
        [HttpGet("GetTransfers")]
        public ApiBaseResponse<List<TransferResponse>> GetTransfers()
        {
            return _memberService.GetTransfers();
        }
        
        [HttpPost("UploadLeaveImage")]
        public async Task<ApiBaseResponse<UploadImageResponse>> UploadLeaveImage(IFormFile request)
        {
            return await _memberService.UploadLeaveImage(new UploadImageRequest(){FormFile = request});
        }
    }
}