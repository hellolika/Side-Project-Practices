using System.ComponentModel;

namespace HRMS.Enum
{
    public enum ApiErrorEnum
    {
        [Description("No Error")]
        NoError,

        [Description("Internal Error")]
        InternalError,

        [Description("Invalid Request Param")]
        InvalidRequest,

        [Description("Member Not Exist")]
        MemberNotFound,

        [Description("Failed to Authorize JWT")]
        AuthorizationFailed,

        [Description("Invalid token")]
        InvalidToken,

        [Description("Incorrect Password")]
        InvalidPassword,

        [Description("Already Do Clock")]
        AlreadyDoClock,

        [Description("Not In Company")]
        NotInCompany,

        [Description("Insufficent Leave Amount")]
        InsufficentLeaveAmount,

        [Description("Date Out Of Range")]
        DateOutOfRange,

        [Description("Request not found, please contact technical supports.")]
        RequestIdNotFound,

        [Description("Request cannot be canceled")]
        CannotCancel,

        [Description("Leave type does not exist")]
        InvalidLeaveType,

        [Description("Already clocked in")]
        AlreadyClockIn,

        [Description("Error! User already clocked out once")]
        AlreadyClockOut,

        [Description("You already requested re-clock for this date")]
        AlreadyReClock,

        [Description("ClockOut time is too close to ClockIn ")]
        DoClockFailed,

        [Description("Incorrect password")]
        IncorrectPassword,

        [Description("New password and confirm password do not match")]
        PasswordsNotMatch,

        [Description("Invalid Model State")]
        InvalidModelState,

        [Description("This request is already approved")]
        RequestAlreadyApproved,

        [Description("This request is already rejected")]
        RequestAlreadyRejected,

        [Description("The leaves requested exceeds leaves remaining")]
        InvalidAmountOfLeaveDays,

        [Description("Cannot request for leaves in different years")]
        DifferentYearsLeaveRequestError,

        [Description("Username already in used, please chooes a new one")]
        UserNameTaken,

        [Description("Email already in used, please make sure you enter the correct email")]
        EmailTaken,

        [Description("This request is already canceled")]
        RequestAlreadyCanceled,

        [Description("Incorrect email or password")]
        LoginFailed,

        [Description("The member requesting to do clock is supposed to be on leave")]
        ClockFailedMemberOnLeave,

        [Description("The dates you requested overlaps with another leave form.")]
        LeaveDatesOverlap,

        [Description("Cannot find announcement with the id provided.")]
        AnnouncementNotFound,

        [Description("User not found.")]
        UserNotFound,

        [Description("Member is still in probation.")]
        MemberStillInProbation,

        [Description("Phone number already taken, please chooes a new one")]
        PhoneNumberTaken,
        
        [Description("Member is already had this role.")]
        MemberRoleExist,
        
        [Description("Role does not exist.")]
        RoleNotFound,
        
        [Description("Member with this role does not exist.")]
        MemberRoleNotExist,

        [Description("Role is already exist.")]
        RoleExist,

        [Description("Invalid Team Id.")]
        InvalidTeamId,

        [Description("This member does not have any leave amount or doesn't exist")]
        MemberHaveNoLeaveAmount,

        [Description("This member already registerd leave amount")]
        AlreadyRegisteredLeaveAmount,
        
        [Description("This member already have monthly transfer.")]
        AlreadyAddMonthly,
        
        [Description("Bank Account is already exist.")]
        BankAccountExist,
        
        [Description("Employee Id is already exist.")]
        EmployeeIdExist,
        
        [Description("Failed to upload image")]
        UploadImageFail,
        
        [Description("Image Size Cannot Bigger than 1MB")]
        ImageSizeLimitMb,
        
        [Description("Permission Denied")]
        InvalidPermission = 403,
        
     
        
        
    }
}